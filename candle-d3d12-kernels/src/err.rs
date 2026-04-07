#[derive(thiserror::Error, Debug)]
pub enum D3D12KernelError {
    #[error("Device creation failed: {0}")]
    DeviceCreation(String),
    #[error("Buffer operation failed: {0}")]
    BufferError(String),
    #[error("Pipeline creation failed: {0}")]
    PipelineCreation(String),
    #[error("Dispatch failed: {0}")]
    Dispatch(String),
    #[error("Data transfer failed: {0}")]
    DataTransfer(String),
}
