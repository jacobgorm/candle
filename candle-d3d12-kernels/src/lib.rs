pub mod err;

#[cfg(target_os = "windows")]
pub mod d3d12;

pub use err::D3D12KernelError;

#[cfg(target_os = "windows")]
pub use d3d12::{BufferBinding, Gpu, GpuBuffer};
#[cfg(target_os = "windows")]
pub use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;
