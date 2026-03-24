// D3D12 compute shader backend for candle.
// This module is only compiled when the `d3d12` feature is enabled.

mod device;

pub use device::D3D12Device;

use crate::backend::{BackendDevice, BackendStorage};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
use candle_d3d12_kernels::{self as d3d12k, BufferBinding, GpuBuffer, Source};
use std::sync::Arc;

#[derive(thiserror::Error, Debug)]
pub enum D3D12Error {
    #[error("{0}")]
    Message(String),
}

impl From<String> for D3D12Error {
    fn from(e: String) -> Self {
        D3D12Error::Message(e)
    }
}

fn d3d12_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Msg(e.to_string())
}

pub struct D3D12Storage {
    pub(crate) device: D3D12Device,
    pub(crate) buffer: Arc<GpuBuffer>,
    pub(crate) count: usize,
    pub(crate) dtype: DType,
}

impl D3D12Storage {
    /// Access the underlying GPU buffer.
    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    /// Access the underlying GPU buffer as an Arc.
    pub fn buffer_arc(&self) -> &Arc<GpuBuffer> {
        &self.buffer
    }

    /// Number of elements in this storage.
    pub fn elem_count(&self) -> usize {
        self.count
    }
}

impl std::fmt::Debug for D3D12Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("D3D12Storage")
            .field("count", &self.count)
            .field("dtype", &self.dtype)
            .finish()
    }
}

// --- Helpers ---

fn unary_entry_point(kernel: &str) -> Option<&'static str> {
    match kernel {
        "uneg" => Some("neg_f32"),
        "uabs" => Some("abs_f32"),
        "uexp" => Some("exp_f32"),
        "ulog" => Some("log_f32"),
        "usin" => Some("sin_f32"),
        "ucos" => Some("cos_f32"),
        "usqrt" => Some("sqrt_f32"),
        "usqr" => Some("sqr_f32"),
        "urecip" => Some("recip_f32"),
        "urelu" => Some("relu_f32"),
        "utanh" => Some("tanh_f32"),
        "ugelu" => Some("gelu_f32"),
        "ugelu_erf" => Some("gelu_erf_f32"),
        "usilu" => Some("silu_f32"),
        "uceil" => Some("ceil_f32"),
        "ufloor" => Some("floor_f32"),
        "uround" => Some("round_f32"),
        _ => None,
    }
}

fn binary_entry_point(kernel: &str) -> Option<&'static str> {
    match kernel {
        "badd" => Some("add_f32"),
        "bsub" => Some("sub_f32"),
        "bmul" => Some("mul_f32"),
        "bdiv" => Some("div_f32"),
        "bmaximum" => Some("max_f32"),
        "bminimum" => Some("min_f32"),
        _ => None,
    }
}

/// Check if sum_dims are the trailing dimensions of a tensor with the given rank.
fn is_trailing_reduction(rank: usize, sum_dims: &[usize]) -> bool {
    if sum_dims.is_empty() {
        return false;
    }
    let mut sorted = sum_dims.to_vec();
    sorted.sort();
    sorted.dedup();
    let n = sorted.len();
    for i in 0..n {
        if sorted[i] != rank - n + i {
            return false;
        }
    }
    true
}

fn bytes_to_cpu_storage(data: &[u8], count: usize, dtype: DType) -> Result<CpuStorage> {
    match dtype {
        DType::F32 => {
            let v: Vec<f32> = data
                .chunks_exact(4)
                .take(count)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(CpuStorage::F32(v))
        }
        DType::F64 => {
            let v: Vec<f64> = data
                .chunks_exact(8)
                .take(count)
                .map(|b| f64::from_le_bytes(b[..8].try_into().unwrap()))
                .collect();
            Ok(CpuStorage::F64(v))
        }
        DType::U8 => Ok(CpuStorage::U8(data[..count].to_vec())),
        DType::U32 => {
            let v: Vec<u32> = data
                .chunks_exact(4)
                .take(count)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(CpuStorage::U32(v))
        }
        DType::I16 => {
            let v: Vec<i16> = data
                .chunks_exact(2)
                .take(count)
                .map(|b| i16::from_le_bytes([b[0], b[1]]))
                .collect();
            Ok(CpuStorage::I16(v))
        }
        DType::I32 => {
            let v: Vec<i32> = data
                .chunks_exact(4)
                .take(count)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Ok(CpuStorage::I32(v))
        }
        DType::I64 => {
            let v: Vec<i64> = data
                .chunks_exact(8)
                .take(count)
                .map(|b| i64::from_le_bytes(b[..8].try_into().unwrap()))
                .collect();
            Ok(CpuStorage::I64(v))
        }
        DType::BF16 => {
            let v: Vec<half::bf16> = data
                .chunks_exact(2)
                .take(count)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]))
                .collect();
            Ok(CpuStorage::BF16(v))
        }
        DType::F16 => {
            let v: Vec<half::f16> = data
                .chunks_exact(2)
                .take(count)
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]))
                .collect();
            Ok(CpuStorage::F16(v))
        }
        dtype => crate::bail!("D3D12 bytes_to_cpu_storage not implemented for {dtype:?}"),
    }
}

// --- D3D12Storage impl ---

impl D3D12Storage {
    /// Upload layout dims+strides as a u32 metadata buffer for strided kernels.
    fn upload_layout_meta(&self, layout: &Layout) -> Result<GpuBuffer> {
        let dims = layout.dims();
        let strides = layout.stride();
        let meta: Vec<u32> = dims
            .iter()
            .map(|&d| d as u32)
            .chain(strides.iter().map(|&s| s as u32))
            .collect();
        let bytes: Vec<u8> = meta.iter().flat_map(|x| x.to_le_bytes()).collect();
        let gpu = &self.device.gpu;
        let buf = gpu.create_buffer(bytes.len() as u64).map_err(d3d12_err)?;
        gpu.upload_to_buffer(&bytes, &buf).map_err(d3d12_err)?;
        Ok(buf)
    }

    /// Get a contiguous F32 buffer for compute operations.
    /// If data is already contiguous at offset 0, returns the existing buffer (cheap Arc clone).
    /// Otherwise creates a GPU-side contiguous copy.
    fn contiguous_f32(&self, layout: &Layout) -> Result<(Arc<GpuBuffer>, usize)> {
        let elem_count = layout.shape().elem_count();
        let gpu = &self.device.gpu;

        if layout.is_contiguous() && layout.start_offset() == 0 {
            return Ok((self.buffer.clone(), elem_count));
        }

        let size_bytes = (elem_count * 4) as u64;
        let new_buf = gpu.create_buffer(size_bytes).map_err(d3d12_err)?;

        if layout.is_contiguous() {
            // Contiguous but with offset - GPU buffer-to-buffer copy
            let src_offset = (layout.start_offset() * self.dtype.size_in_bytes()) as u64;
            gpu.copy_buffer_region(&self.buffer, src_offset, &new_buf, 0, size_bytes)
                .map_err(d3d12_err)?;
        } else {
            // Strided - use copy_strided_f32 kernel with offset SRV
            let meta = self.upload_layout_meta(layout)?;
            let pipelines = &self.device.pipelines;
            let pso = pipelines
                .load_pipeline(gpu, Source::Indexing, "copy_strided_f32")
                .map_err(d3d12_err)?;

            let count = elem_count as u32;
            let num_dims = layout.dims().len() as u32;
            let constants = [count, num_dims, 0u32]; // dst_offset = 0

            let src_remaining = self.count.saturating_sub(layout.start_offset());
            let srvs = [
                BufferBinding::structured_f32_offset(
                    &self.buffer,
                    layout.start_offset() as u32,
                    src_remaining as u32,
                ),
                BufferBinding::raw(&meta),
            ];
            let uav = BufferBinding::structured_f32(&new_buf, count);
            let groups = d3d12k::utils::linear_groups(count);

            gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
                .map_err(d3d12_err)?;
        }

        Ok((Arc::new(new_buf), elem_count))
    }

    fn new_storage(&self, buffer: GpuBuffer, count: usize, dtype: DType) -> D3D12Storage {
        D3D12Storage {
            device: self.device.clone(),
            buffer: Arc::new(buffer),
            count,
            dtype,
        }
    }
}

// --- BackendStorage ---

impl BackendStorage for D3D12Storage {
    type Device = D3D12Device;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let elem_count = layout.shape().elem_count();
        let gpu = &self.device.gpu;

        if layout.is_contiguous() && layout.start_offset() == 0 && elem_count == self.count {
            // Full buffer clone via GPU copy
            let size_bytes = (self.count * self.dtype.size_in_bytes()) as u64;
            let new_buf = gpu.create_buffer(size_bytes).map_err(d3d12_err)?;
            gpu.copy_buffer_region(&self.buffer, 0, &new_buf, 0, size_bytes)
                .map_err(d3d12_err)?;
            return Ok(self.new_storage(new_buf, self.count, self.dtype));
        }

        if self.dtype == DType::F32 {
            let (buf, count) = self.contiguous_f32(layout)?;
            // If contiguous_f32 returned the same Arc, we need to actually copy
            if Arc::ptr_eq(&buf, &self.buffer) {
                let size_bytes = (count * 4) as u64;
                let new_buf = gpu.create_buffer(size_bytes).map_err(d3d12_err)?;
                gpu.copy_buffer_region(&self.buffer, 0, &new_buf, 0, size_bytes)
                    .map_err(d3d12_err)?;
                return Ok(self.new_storage(new_buf, count, self.dtype));
            }
            return Ok(D3D12Storage {
                device: self.device.clone(),
                buffer: buf,
                count,
                dtype: self.dtype,
            });
        }

        // Non-F32: fall back to CPU roundtrip
        let cpu = self.to_cpu_storage()?;
        let cpu_result = cpu.try_clone(layout)?;
        self.device.storage_from_cpu_storage(&cpu_result)
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let size_bytes = (self.count * self.dtype.size_in_bytes()) as u64;
        let data = self
            .device
            .gpu
            .download_buffer(&self.buffer, size_bytes)
            .map_err(d3d12_err)?;
        bytes_to_cpu_storage(&data, self.count, self.dtype)
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype != DType::F32 {
            let cpu = self.to_cpu_storage()?;
            let result = cpu.affine(layout, mul, add)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let (input_buf, elem_count) = self.contiguous_f32(layout)?;
        let output_buf = self
            .device
            .gpu
            .create_buffer((elem_count * 4) as u64)
            .map_err(d3d12_err)?;

        d3d12k::call_affine(
            &self.device.gpu,
            &self.device.pipelines,
            elem_count as u32,
            &input_buf,
            &output_buf,
            mul as f32,
            add as f32,
        )
        .map_err(d3d12_err)?;

        Ok(self.new_storage(output_buf, elem_count, DType::F32))
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.powf(layout, e)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.elu(layout, alpha)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        if self.dtype != DType::F32 {
            let cpu = self.to_cpu_storage()?;
            let result = cpu.reduce_op(op, layout, sum_dims)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let src_dims = layout.shape().dims();
        let rank = src_dims.len();
        let total = layout.shape().elem_count();

        // Only handle trailing reduction on contiguous data on GPU
        if !is_trailing_reduction(rank, sum_dims) || !layout.is_contiguous() {
            let cpu = self.to_cpu_storage()?;
            let result = cpu.reduce_op(op, layout, sum_dims)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let (input_buf, _) = self.contiguous_f32(layout)?;

        let reduction_size: usize = sum_dims.iter().map(|&d| src_dims[d]).product();
        let out_length = total / reduction_size;

        match op {
            ReduceOp::ArgMax | ReduceOp::ArgMin => {
                let output_buf = self
                    .device
                    .gpu
                    .create_buffer((out_length * 4) as u64)
                    .map_err(d3d12_err)?;
                let entry = match op {
                    ReduceOp::ArgMax => "argmax_f32",
                    ReduceOp::ArgMin => "argmin_f32",
                    _ => unreachable!(),
                };
                d3d12k::call_arg_reduce(
                    &self.device.gpu,
                    &self.device.pipelines,
                    entry,
                    total as u32,
                    out_length as u32,
                    &input_buf,
                    &output_buf,
                )
                .map_err(d3d12_err)?;
                Ok(self.new_storage(output_buf, out_length, DType::U32))
            }
            _ => {
                let output_buf = self
                    .device
                    .gpu
                    .create_buffer((out_length * 4) as u64)
                    .map_err(d3d12_err)?;
                let entry = match op {
                    ReduceOp::Sum => "sum_f32",
                    ReduceOp::Min => "min_f32",
                    ReduceOp::Max => "max_f32",
                    _ => unreachable!(),
                };
                d3d12k::call_reduce(
                    &self.device.gpu,
                    &self.device.pipelines,
                    entry,
                    total as u32,
                    out_length as u32,
                    &input_buf,
                    &output_buf,
                )
                .map_err(d3d12_err)?;
                Ok(self.new_storage(output_buf, out_length, DType::F32))
            }
        }
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let lhs_cpu = self.to_cpu_storage()?;
        let rhs_cpu = rhs.to_cpu_storage()?;
        let result = lhs_cpu.cmp(op, &rhs_cpu, lhs_l, rhs_l)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.to_dtype(layout, dtype)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let entry = unary_entry_point(B::KERNEL);
        if self.dtype != DType::F32 || entry.is_none() {
            let cpu = self.to_cpu_storage()?;
            let result = cpu.unary_impl::<B>(layout)?;
            return self.device.storage_from_cpu_storage(&result);
        }
        let entry = entry.unwrap();

        let (input_buf, elem_count) = self.contiguous_f32(layout)?;
        let output_buf = self
            .device
            .gpu
            .create_buffer((elem_count * 4) as u64)
            .map_err(d3d12_err)?;

        d3d12k::call_unary_contiguous(
            &self.device.gpu,
            &self.device.pipelines,
            entry,
            elem_count as u32,
            &input_buf,
            &output_buf,
        )
        .map_err(d3d12_err)?;

        Ok(self.new_storage(output_buf, elem_count, DType::F32))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let entry = binary_entry_point(B::KERNEL);
        if self.dtype != DType::F32 || entry.is_none() {
            let lhs_cpu = self.to_cpu_storage()?;
            let rhs_cpu = rhs.to_cpu_storage()?;
            let result = lhs_cpu.binary_impl::<B>(&rhs_cpu, lhs_l, rhs_l)?;
            return self.device.storage_from_cpu_storage(&result);
        }
        let entry = entry.unwrap();
        let elem_count = lhs_l.shape().elem_count();

        let (lhs_buf, _) = self.contiguous_f32(lhs_l)?;
        let (rhs_buf, _) = rhs.contiguous_f32(rhs_l)?;

        let output_buf = self
            .device
            .gpu
            .create_buffer((elem_count * 4) as u64)
            .map_err(d3d12_err)?;

        d3d12k::call_binary_contiguous(
            &self.device.gpu,
            &self.device.pipelines,
            entry,
            elem_count as u32,
            &lhs_buf,
            &rhs_buf,
            &output_buf,
        )
        .map_err(d3d12_err)?;

        Ok(self.new_storage(output_buf, elem_count, DType::F32))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let cond_cpu = self.to_cpu_storage()?;
        let t_cpu = t.to_cpu_storage()?;
        let f_cpu = f.to_cpu_storage()?;
        let result = cond_cpu.where_cond(layout, &t_cpu, t_l, &f_cpu, f_l)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        if self.dtype != DType::F32
            || !layout.is_contiguous()
            || layout.start_offset() != 0
            || !kernel_l.is_contiguous()
            || kernel_l.start_offset() != 0
        {
            let i_cpu = self.to_cpu_storage()?;
            let k_cpu = kernel.to_cpu_storage()?;
            let result = i_cpu.conv1d(layout, &k_cpu, kernel_l, params)?;
            return self.device.storage_from_cpu_storage(&result);
        }

        let b = params.b_size;
        let c_in = params.c_in;
        let c_out = params.c_out;
        let l_in = params.l_in;
        let l_out = params.l_out();
        let k_size = params.k_size;
        let stride = params.stride;
        let padding = params.padding;
        let dilation = params.dilation;

        let out_count = b * c_out * l_out;
        let output_buf = self
            .device
            .gpu
            .create_buffer((out_count * 4) as u64)
            .map_err(d3d12_err)?;

        d3d12k::call_conv1d(
            &self.device.gpu,
            &self.device.pipelines,
            b as u32,
            c_in as u32,
            c_out as u32,
            l_in as u32,
            l_out as u32,
            k_size as u32,
            stride as u32,
            padding as u32,
            dilation as u32,
            &self.buffer,
            self.count as u32,
            &kernel.buffer,
            kernel.count as u32,
            &output_buf,
        )
        .map_err(d3d12_err)?;

        Ok(self.new_storage(output_buf, out_count, DType::F32))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        let i_cpu = self.to_cpu_storage()?;
        let k_cpu = kernel.to_cpu_storage()?;
        let result = i_cpu.conv_transpose1d(layout, &k_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let i_cpu = self.to_cpu_storage()?;
        let k_cpu = kernel.to_cpu_storage()?;
        let result = i_cpu.conv2d(layout, &k_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let i_cpu = self.to_cpu_storage()?;
        let k_cpu = kernel.to_cpu_storage()?;
        let result = i_cpu.conv_transpose2d(layout, &k_cpu, kernel_l, params)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn index_select(
        &self,
        ids: &Self,
        src_l: &Layout,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        // GPU path for contiguous F32 source with U32 indices
        if self.dtype == DType::F32
            && ids.dtype == DType::U32
            && src_l.is_contiguous()
            && src_l.start_offset() == 0
            && ids_l.is_contiguous()
            && ids_l.start_offset() == 0
        {
            let src_dims = src_l.dims();
            let left_size: usize = src_dims[..dim].iter().product();
            let dim_size = src_dims[dim];
            let right_size: usize = src_dims[dim + 1..].iter().product();
            let ids_count = ids_l.shape().elem_count();
            let out_count = left_size * ids_count * right_size;

            let output_buf = self
                .device
                .gpu
                .create_buffer((out_count * 4) as u64)
                .map_err(d3d12_err)?;

            d3d12k::call_index_select(
                &self.device.gpu,
                &self.device.pipelines,
                out_count as u32,
                left_size as u32,
                dim_size as u32,
                right_size as u32,
                &self.buffer,
                self.count as u32,
                &ids.buffer,
                ids_count as u32,
                &output_buf,
            )
            .map_err(d3d12_err)?;

            return Ok(self.new_storage(output_buf, out_count, DType::F32));
        }

        // CPU fallback
        let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.index_select(&ids_cpu, src_l, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn gather(
        &self,
        src_l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.gather(src_l, &ids_cpu, ids_l, dim)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        dst_cpu.scatter_set(layout, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
        let new_storage = self.device.storage_from_cpu_storage(&dst_cpu)?;
        *self = new_storage;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        dst_cpu.scatter_add_set(layout, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
        let new_storage = self.device.storage_from_cpu_storage(&dst_cpu)?;
        *self = new_storage;
        Ok(())
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let dst_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        let result = dst_cpu.index_add(layout, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        if self.dtype != DType::F32 {
            crate::bail!("D3D12 matmul only supports F32, got {:?}", self.dtype);
        }

        let out_count = b * m * n;
        let output_buf = self
            .device
            .gpu
            .create_buffer((out_count * 4) as u64)
            .map_err(d3d12_err)?;

        // Extract strides: lhs is (b, m, k), rhs is (b, k, n)
        let lhs_s = lhs_l.stride();
        let (lhs_b_stride, lhs_m_stride, lhs_k_stride) = match lhs_s.len() {
            3 => (lhs_s[0], lhs_s[1], lhs_s[2]),
            2 => (m * k, lhs_s[0], lhs_s[1]),
            _ => (m * k, k, 1),
        };

        let rhs_s = rhs_l.stride();
        let (rhs_b_stride, rhs_k_stride, rhs_n_stride) = match rhs_s.len() {
            3 => (rhs_s[0], rhs_s[1], rhs_s[2]),
            2 => (k * n, rhs_s[0], rhs_s[1]),
            _ => (k * n, n, 1),
        };

        let gpu = &self.device.gpu;
        let pipelines = &self.device.pipelines;
        let pso = pipelines
            .load_pipeline(gpu, Source::Matmul, "matmul_f32")
            .map_err(d3d12_err)?;

        let constants = [
            b as u32,
            m as u32,
            n as u32,
            k as u32,
            lhs_b_stride as u32,
            lhs_m_stride as u32,
            lhs_k_stride as u32,
            rhs_b_stride as u32,
            rhs_k_stride as u32,
            rhs_n_stride as u32,
        ];

        // Use offset SRVs to handle start_offset
        let lhs_remaining = self.count.saturating_sub(lhs_l.start_offset());
        let rhs_remaining = rhs.count.saturating_sub(rhs_l.start_offset());

        let srvs = [
            BufferBinding::structured_f32_offset(
                &self.buffer,
                lhs_l.start_offset() as u32,
                lhs_remaining as u32,
            ),
            BufferBinding::structured_f32_offset(
                &rhs.buffer,
                rhs_l.start_offset() as u32,
                rhs_remaining as u32,
            ),
        ];

        let uav = BufferBinding::structured_f32(&output_buf, out_count as u32);
        let groups = d3d12k::utils::matmul_groups(n as u32, m as u32, b as u32);

        gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
            .map_err(d3d12_err)?;

        Ok(self.new_storage(output_buf, out_count, DType::F32))
    }

    fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        let elem_count = src_l.shape().elem_count();
        let gpu = &self.device.gpu;

        if self.dtype == DType::F32 {
            if src_l.is_contiguous() {
                // Fast path: GPU buffer-to-buffer copy
                let src_offset_bytes =
                    (src_l.start_offset() * self.dtype.size_in_bytes()) as u64;
                let dst_offset_bytes = (dst_offset * dst.dtype.size_in_bytes()) as u64;
                let size_bytes = (elem_count * self.dtype.size_in_bytes()) as u64;
                gpu.copy_buffer_region(
                    &self.buffer,
                    src_offset_bytes,
                    &dst.buffer,
                    dst_offset_bytes,
                    size_bytes,
                )
                .map_err(d3d12_err)?;
                return Ok(());
            }

            // Strided path: use copy_strided_f32 kernel
            let meta = self.upload_layout_meta(src_l)?;
            let pipelines = &self.device.pipelines;
            let pso = pipelines
                .load_pipeline(gpu, Source::Indexing, "copy_strided_f32")
                .map_err(d3d12_err)?;

            let count = elem_count as u32;
            let num_dims = src_l.dims().len() as u32;
            let constants = [count, num_dims, dst_offset as u32];

            let src_remaining = self.count.saturating_sub(src_l.start_offset());
            let srvs = [
                BufferBinding::structured_f32_offset(
                    &self.buffer,
                    src_l.start_offset() as u32,
                    src_remaining as u32,
                ),
                BufferBinding::raw(&meta),
            ];
            let uav = BufferBinding::structured_f32(&dst.buffer, dst.count as u32);
            let groups = d3d12k::utils::linear_groups(count);

            gpu.dispatch(&pso, &constants, &srvs, &uav, groups)
                .map_err(d3d12_err)?;
            return Ok(());
        }

        // Non-F32: CPU fallback
        let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let new_dst = self.device.storage_from_cpu_storage(&dst_cpu)?;
        *dst = new_dst;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        if self.dtype == DType::F32 {
            d3d12k::call_copy2d(
                &self.device.gpu,
                &self.device.pipelines,
                d1 as u32,
                d2 as u32,
                src_stride1 as u32,
                dst_stride1 as u32,
                src_offset as u32,
                dst_offset as u32,
                &self.buffer,
                self.count as u32,
                &dst.buffer,
                dst.count as u32,
            )
            .map_err(d3d12_err)?;
            return Ok(());
        }

        // Non-F32: CPU fallback
        let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy2d(
            &mut dst_cpu,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )?;
        let new_dst = self.device.storage_from_cpu_storage(&dst_cpu)?;
        *dst = new_dst;
        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        if !layout.is_contiguous() {
            crate::bail!("D3D12 const_set only supports contiguous layouts");
        }

        let elem_count = layout.shape().elem_count();
        let val_f32: f32 = match v {
            crate::scalar::Scalar::F32(v) => v,
            crate::scalar::Scalar::F64(v) => v as f32,
            crate::scalar::Scalar::U32(v) => v as f32,
            crate::scalar::Scalar::U8(v) => v as f32,
            crate::scalar::Scalar::I64(v) => v as f32,
            crate::scalar::Scalar::I32(v) => v as f32,
            crate::scalar::Scalar::I16(v) => v as f32,
            crate::scalar::Scalar::BF16(v) => v.to_f32(),
            crate::scalar::Scalar::F16(v) => v.to_f32(),
            _ => crate::bail!("D3D12 const_set: unsupported scalar type"),
        };

        d3d12k::call_fill_f32(
            &self.device.gpu,
            &self.device.pipelines,
            elem_count as u32,
            val_f32,
            &self.buffer,
        )
        .map_err(d3d12_err)?;

        Ok(())
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.avg_pool2d(layout, kernel_size, stride)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.max_pool2d(layout, kernel_size, stride)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn upsample_nearest1d(&self, layout: &Layout, target_size: usize) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.upsample_nearest1d(layout, target_size)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn upsample_nearest2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result = cpu.upsample_nearest2d(layout, target_h, target_w)?;
        self.device.storage_from_cpu_storage(&result)
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        target_h: usize,
        target_w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        let cpu = self.to_cpu_storage()?;
        let result =
            cpu.upsample_bilinear2d(layout, target_h, target_w, align_corners, scale_h, scale_w)?;
        self.device.storage_from_cpu_storage(&result)
    }
}
