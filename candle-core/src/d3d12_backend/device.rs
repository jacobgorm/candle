use crate::backend::{BackendDevice, BackendStorage};
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape, WithDType};
use candle_d3d12_kernels::{Gpu, Pipelines};
use std::sync::Arc;

use super::D3D12Storage;

static D3D12_DEVICE_ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct D3D12Device {
    id: usize,
    ordinal: usize,
    pub(crate) gpu: Arc<Gpu>,
    pub(crate) pipelines: Arc<Pipelines>,
    seed: Arc<std::sync::Mutex<u64>>,
}

impl D3D12Device {
    /// Access the underlying D3D12 GPU context.
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// Access the pipeline cache.
    pub fn pipelines(&self) -> &Arc<Pipelines> {
        &self.pipelines
    }
}

impl BackendDevice for D3D12Device {
    type Storage = D3D12Storage;

    fn new(ordinal: usize) -> Result<Self> {
        let id = D3D12_DEVICE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let gpu = Gpu::new(ordinal).map_err(|e| crate::Error::Msg(e.to_string()))?;
        Ok(Self {
            id,
            ordinal,
            gpu: Arc::new(gpu),
            pipelines: Arc::new(Pipelines::new()),
            seed: Arc::new(std::sync::Mutex::new(299792458)),
        })
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::D3D12 {
            gpu_id: self.ordinal,
        }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_bytes = (elem_count * dtype.size_in_bytes()) as u64;
        let buffer = self
            .gpu
            .create_buffer(size_bytes)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        // Zero-fill via upload
        let zeros = vec![0u8; size_bytes as usize];
        self.gpu
            .upload_to_buffer(&zeros, &buffer)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        Ok(D3D12Storage {
            device: self.clone(),
            buffer: Arc::new(buffer),
            count: elem_count,
            dtype,
        })
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_bytes = (elem_count * dtype.size_in_bytes()) as u64;
        let buffer = self
            .gpu
            .create_buffer(size_bytes)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        Ok(D3D12Storage {
            device: self.clone(),
            buffer: Arc::new(buffer),
            count: elem_count,
            dtype,
        })
    }

    fn storage_from_slice<T: WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let cpu_storage = T::to_cpu_storage(data);
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let dtype = storage.dtype();
        let bytes = cpu_storage_to_bytes(storage);
        let elem_count = cpu_storage_elem_count(storage);
        let size_bytes = bytes.len() as u64;
        let buffer = self
            .gpu
            .create_buffer(size_bytes)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        self.gpu
            .upload_to_buffer(&bytes, &buffer)
            .map_err(|e| crate::Error::Msg(e.to_string()))?;
        Ok(D3D12Storage {
            device: self.clone(),
            buffer: Arc::new(buffer),
            count: elem_count,
            dtype,
        })
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        lo: f64,
        up: f64,
    ) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice;
        let cpu_storage = cpu.rand_uniform(shape, dtype, lo, up)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        std: f64,
    ) -> Result<Self::Storage> {
        let cpu = crate::cpu_backend::CpuDevice;
        let cpu_storage = cpu.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock().unwrap() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed.lock().unwrap())
    }

    fn synchronize(&self) -> Result<()> {
        // Each dispatch already waits for completion via fence
        Ok(())
    }
}

fn cpu_storage_to_bytes(storage: &CpuStorage) -> Vec<u8> {
    match storage {
        CpuStorage::U8(v) => v.clone(),
        CpuStorage::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::BF16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        CpuStorage::F64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        _ => Vec::new(),
    }
}

fn cpu_storage_elem_count(storage: &CpuStorage) -> usize {
    match storage {
        CpuStorage::U8(v) => v.len(),
        CpuStorage::U32(v) => v.len(),
        CpuStorage::I16(v) => v.len(),
        CpuStorage::I32(v) => v.len(),
        CpuStorage::I64(v) => v.len(),
        CpuStorage::BF16(v) => v.len(),
        CpuStorage::F16(v) => v.len(),
        CpuStorage::F32(v) => v.len(),
        CpuStorage::F64(v) => v.len(),
        _ => 0,
    }
}
