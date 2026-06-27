#include "cusparselt_linear.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

__global__ void pack_transpose_kernel(const half* __restrict__ input,
                                      half* __restrict__ packed,
                                      int actual_n,
                                      int padded_n,
                                      int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = k * padded_n;
  if (idx >= total) return;
  int kk = idx / padded_n;
  int n = idx - kk * padded_n;
  packed[idx] = (n < actual_n) ? input[n * k + kk] : __float2half(0.0f);
}

__global__ void unpack_transpose_bias_kernel(const half* __restrict__ d,
                                             const half* __restrict__ bias,
                                             half* __restrict__ output,
                                             int actual_n,
                                             int padded_n,
                                             int m,
                                             bool has_bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = actual_n * m;
  if (idx >= total) return;
  int n = idx / m;
  int mm = idx - n * m;
  half v = d[mm * padded_n + n];
  if (has_bias) {
    v = __hadd(v, bias[mm]);
  }
  output[idx] = v;
}

}  // namespace

void pack_transpose_fp16(torch::Tensor input_2d, torch::Tensor packed_kxn, int64_t actual_n) {
  const int k = static_cast<int>(input_2d.size(1));
  const int padded_n = static_cast<int>(packed_kxn.size(1));
  const int total = k * padded_n;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pack_transpose_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(input_2d.data_ptr<at::Half>()),
      reinterpret_cast<half*>(packed_kxn.data_ptr<at::Half>()),
      static_cast<int>(actual_n),
      padded_n,
      k);
}

void unpack_transpose_bias_fp16(torch::Tensor d_mxn, torch::Tensor bias, torch::Tensor output_2d, int64_t actual_n, bool has_bias) {
  const int m = static_cast<int>(d_mxn.size(0));
  const int padded_n = static_cast<int>(d_mxn.size(1));
  const int total = static_cast<int>(actual_n) * m;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const half* bias_ptr = has_bias ? reinterpret_cast<const half*>(bias.data_ptr<at::Half>()) : nullptr;
  unpack_transpose_bias_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half*>(d_mxn.data_ptr<at::Half>()),
      bias_ptr,
      reinterpret_cast<half*>(output_2d.data_ptr<at::Half>()),
      static_cast<int>(actual_n),
      padded_n,
      m,
      has_bias);
}

