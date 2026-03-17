use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, RecordBatch, UInt64Array};
use arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::physical_expr::Distribution;
use datafusion::physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, RecordBatchStream, SendableRecordBatchStream};
use futures::{Stream, StreamExt};

use crate::operations::merge::{TARGET_DELETE_COLUMN, TARGET_ROW_INDEX_COLUMN, TARGET_UPDATE_COLUMN};
use crate::DeltaTableError;

#[derive(Debug)]
pub(crate) struct MergeValidationExec {
    input: Arc<dyn ExecutionPlan>,
    row_index_expr: Arc<dyn PhysicalExpr>,
}

impl MergeValidationExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            row_index_expr: expr
        }
    }
}

impl ExecutionPlan for MergeValidationExec {
    fn name(&self) -> &str {
        Self::static_name()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.input.schema()
    }

    fn properties(&self) -> &datafusion::physical_plan::PlanProperties {
        self.input.properties()
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::HashPartitioned(vec![self.row_index_expr.clone()]); 1]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Plan(
                "MergeValidationExec wrong number of children".to_string(),
            ));
        }
        Ok(Arc::new(Self::new(
            children[0].clone(),
            self.row_index_expr.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<datafusion::execution::TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        Ok(Box::pin(MergeValidationStream::new(input, self.schema())))
    }
}

impl DisplayAs for MergeValidationExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default
            | DisplayFormatType::Verbose
            | DisplayFormatType::TreeRender => {
                write!(f, "MergeValidation")?;
                Ok(())
            }
        }
    }
}

struct MergeValidationStream {
    schema: SchemaRef,
    input: SendableRecordBatchStream,
    target_matches: HashMap<u64, bool>,
}

impl MergeValidationStream {
    fn new(input: SendableRecordBatchStream, schema: SchemaRef) -> Self {
        Self {
            schema,
            input,
            target_matches: HashMap::new(),
        }
    }

    fn validate_batch(&mut self, batch: &RecordBatch) -> DataFusionResult<()> {
        let target_row_index = batch
            .column_by_name(TARGET_ROW_INDEX_COLUMN)
            .ok_or_else(required_operation_column_err)?;
        let target_update = batch
            .column_by_name(TARGET_UPDATE_COLUMN)
            .ok_or_else(required_operation_column_err)?;
        let target_delete = batch
            .column_by_name(TARGET_DELETE_COLUMN)
            .ok_or_else(required_operation_column_err)?;

        let target_row_index = target_row_index
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::External(Box::new(DeltaTableError::Generic(
                    "Merge target row index column is not UInt64".to_string(),
                )))
            })?;

        for row in 0..batch.num_rows() {
            if target_row_index.is_null(row) {
                continue;
            }

            let row_idx = target_row_index.value(row);
            let is_delete = target_delete.is_null(row);
            let is_update = target_update.is_null(row);

            //TODO verify this logic and add test cases for other scenarios
            match self.target_matches.entry(row_idx) {
                Entry::Vacant(v) => {
                    v.insert(is_delete);
                }
                Entry::Occupied(mut o) => {
                    let seen_all_deletes = *o.get();

                    if is_update || !is_delete || !seen_all_deletes {
                        return Err(DataFusionError::External(Box::new(DeltaTableError::Generic(
                            "Merge matched a single target row with multiple source rows".to_string(),
                        ))));
                    }

                    o.insert(true);
                }
            }
        }

        Ok(())
    }
}

fn required_operation_column_err() -> DataFusionError {
    DataFusionError::External(Box::new(DeltaTableError::Generic(
        "Required operation column is missing".to_string(),
    )))
}

impl Stream for MergeValidationStream {
    type Item = DataFusionResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.input.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                if let Err(err) = self.validate_batch(&batch) {
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(Some(Ok(batch)))
            }
            Poll::Ready(Some(Err(err))) => Poll::Ready(Some(Err(err))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.input.size_hint()
    }
}

impl RecordBatchStream for MergeValidationStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

#[derive(Debug, Hash, Eq, PartialEq, PartialOrd)]
pub(crate) struct MergeValidation {
    pub input: LogicalPlan,
    pub expr: Expr
}

impl UserDefinedLogicalNodeCore for MergeValidation {
    fn name(&self) -> &str {
        "MergeValidation"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &datafusion::common::DFSchemaRef {
        self.input.schema()
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![self.expr.clone()]
    }

    fn fmt_for_explain(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "MergeValidation")
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<datafusion::logical_expr::Expr>,
        inputs: Vec<LogicalPlan>,
    ) -> DataFusionResult<Self> {
        Ok(Self {
            input: inputs[0].clone(),
            expr: exprs[0].clone(),
        })
    }
}


#[cfg(test)]
mod tests {
    use super::MergeValidationStream;
    use crate::operations::merge::{
        TARGET_DELETE_COLUMN, TARGET_ROW_INDEX_COLUMN, TARGET_UPDATE_COLUMN,
    };
    use arrow::array::{BooleanArray, RecordBatch, UInt64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::stream;
    use std::sync::Arc;

    fn validation_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(TARGET_ROW_INDEX_COLUMN, DataType::UInt64, true),
            Field::new(TARGET_UPDATE_COLUMN, DataType::Boolean, true),
            Field::new(TARGET_DELETE_COLUMN, DataType::Boolean, true),
        ]))
    }

    fn validation_stream() -> MergeValidationStream {
        let schema = validation_schema();
        let input = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::empty::<datafusion::common::Result<RecordBatch>>(),
        ));

        MergeValidationStream::new(input, schema)
    }

    fn matched_batch(update_indices: Vec<u64>, delete_indices: Vec<u64>) -> RecordBatch {
        let schema = validation_schema();
        let updates: Vec<Option<bool>> = vec![None; update_indices.len()]
            .into_iter()
            .chain(vec![Some(false); delete_indices.len()].into_iter())
            .collect();
        let deletes: Vec<Option<bool>> = vec![Some(false); update_indices.len()]
            .into_iter()
            .chain(vec![None; delete_indices.len()].into_iter())
            .collect();

        let all_indices: Vec<Option<u64>> = update_indices
            .into_iter()
            .chain(delete_indices.into_iter())
            .map(Some)
            .collect();

        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(all_indices)),
                Arc::new(BooleanArray::from(updates)),
                Arc::new(BooleanArray::from(deletes)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_validation_distinct() {
        let mut validation = validation_stream();
        let first_batch = matched_batch(vec![1, 2], vec![3, 4]);
        let second_batch = matched_batch(vec![5, 6], vec![7, 8]);

        validation.validate_batch(&first_batch).unwrap();
        validation.validate_batch(&second_batch).unwrap();
    }

    #[test]
    fn test_validation_duplicate_updates() {
        let mut validation = validation_stream();
        let first_batch = matched_batch(vec![1, 2], vec![4]);
        let second_batch = matched_batch(vec![2, 3], vec![5]);

        validation.validate_batch(&first_batch).unwrap();
        let _err = validation
            .validate_batch(&second_batch)
            .expect_err("expected duplicate target row to fail validation");
    }

    #[test]
    fn test_validation_duplicate_delete_and_other() {
        let mut validation = validation_stream();

        let batch = RecordBatch::try_new(
            validation_schema(),
            vec![
                Arc::new(UInt64Array::from(vec![Some(42_u64), Some(42_u64)])),
                Arc::new(BooleanArray::from(vec![Some(false), Some(false)])),
                Arc::new(BooleanArray::from(vec![None, Some(false)])),
            ],
        )
        .unwrap();

        validation
            .validate_batch(&batch)
            .expect_err("expected duplicate target row to fail validation");
    }
}
