use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, BooleanArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{Expr, LogicalPlan, UserDefinedLogicalNodeCore};
use datafusion::physical_expr::Distribution;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PhysicalExpr, RecordBatchStream,
    SendableRecordBatchStream,
};
use futures::{Stream, StreamExt};

use crate::DeltaTableError;
use crate::operations::merge::TARGET_DUPLICATE_MATCH_VIOLATION_COLUMN;

#[derive(Debug)]
pub(crate) struct MergeValidationExec {
    input: Arc<dyn ExecutionPlan>,
    row_index_expr: Arc<dyn PhysicalExpr>,
}

impl MergeValidationExec {
    pub fn new(input: Arc<dyn ExecutionPlan>, expr: Arc<dyn PhysicalExpr>) -> Self {
        Self {
            input,
            row_index_expr: expr,
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
}

impl MergeValidationStream {
    fn new(input: SendableRecordBatchStream, schema: SchemaRef) -> Self {
        Self { schema, input }
    }

    fn validate_batch(&self, batch: &RecordBatch) -> DataFusionResult<()> {
        let violation_column = batch
            .column_by_name(TARGET_DUPLICATE_MATCH_VIOLATION_COLUMN)
            .ok_or_else(required_operation_column_err)?;

        let violation_array = violation_column
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::External(Box::new(DeltaTableError::Generic(
                    "Merge duplicate match violation column is not Boolean".to_string(),
                )))
            })?;

        // Check if any row has a violation (true value)
        for row in 0..batch.num_rows() {
            if !violation_array.is_null(row) && violation_array.value(row) {
                return Err(DataFusionError::External(Box::new(
                    DeltaTableError::Generic(
                        "Merge matched a single target row with multiple source rows".to_string(),
                    ),
                )));
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
    pub expr: Expr,
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
    // Window-based validation is constructed during merge plan building via datafusion expressions,
    // not during execution. Tests have been moved to mod.rs where the logical plan is constructed.
}
