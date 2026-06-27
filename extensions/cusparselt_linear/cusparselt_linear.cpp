#include "cusparselt_linear.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusparseLt.h>

#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA_THROW(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e)); }} while (0)
#define CHECK_CUSPARSE_THROW(call) do { cusparseStatus_t s = (call); if (s != CUSPARSE_STATUS_SUCCESS) { \
  throw std::runtime_error("cuSPARSELt error status=" + std::to_string((int)s)); }} while (0)

namespace {

int64_t padded_n8(int64_t n) {
  return ((n + 7) / 8) * 8;
}

struct LtPlanCache {
  alignas(16) cusparseLtMatDescriptor_t matA{}, matB{}, matC{};
  alignas(16) cusparseLtMatmulDescriptor_t matmul{};
  alignas(16) cusparseLtMatmulAlgSelection_t alg{};
  alignas(16) cusparseLtMatmulPlan_t plan{};
  int n = 0;
  torch::Tensor compressed;
  torch::Tensor compress_buffer;
  torch::Tensor workspace;
  size_t compressed_size = 0;
  size_t compress_buffer_size = 0;
  size_t workspace_size = 0;
  bool valid = false;

  ~LtPlanCache() {
    if (valid) {
      cusparseLtMatmulPlanDestroy(&plan);
      cusparseLtMatDescriptorDestroy(&matA);
      cusparseLtMatDescriptorDestroy(&matB);
      cusparseLtMatDescriptorDestroy(&matC);
    }
  }
};

}  // namespace

class CuSparseLtLinearCpp {
 public:
  CuSparseLtLinearCpp(torch::Tensor weight_2to4, torch::Tensor bias) {
    if (!weight_2to4.is_cuda() || weight_2to4.scalar_type() != torch::kFloat16 || weight_2to4.dim() != 2) {
      throw std::runtime_error("weight must be CUDA fp16 [Dout, Din]");
    }
    weight_ = weight_2to4.contiguous();
    has_bias_ = bias.defined() && bias.numel() > 0;
    if (has_bias_) {
      if (!bias.is_cuda() || bias.scalar_type() != torch::kFloat16 || bias.dim() != 1) {
        throw std::runtime_error("bias must be CUDA fp16 [Dout]");
      }
      bias_ = bias.contiguous();
    } else {
      bias_ = torch::empty({0}, weight_.options());
    }
    m_ = static_cast<int>(weight_.size(0));
    k_ = static_cast<int>(weight_.size(1));
    if (has_bias_ && bias_.size(0) != m_) {
      throw std::runtime_error("bias Dout mismatch");
    }
    CHECK_CUSPARSE_THROW(cusparseLtInit(&handle_));
  }

  ~CuSparseLtLinearCpp() {
    caches_.clear();
    cusparseLtDestroy(&handle_);
  }

  torch::Tensor forward(torch::Tensor input) {
    if (!input.is_cuda() || input.scalar_type() != torch::kFloat16) {
      throw std::runtime_error("input must be CUDA fp16");
    }
    if (input.dim() != 2 && input.dim() != 3) {
      throw std::runtime_error("input must be [N, Din] or [B, T, Din]");
    }
    if (input.size(input.dim() - 1) != k_) {
      throw std::runtime_error("input Din mismatch");
    }
    std::vector<int64_t> out_shape;
    if (input.dim() == 2) {
      out_shape = {input.size(0), m_};
    } else {
      out_shape = {input.size(0), input.size(1), m_};
    }
    const int64_t actual_n = input.numel() / k_;
    const int64_t padded_n = padded_n8(actual_n);
    auto cache = prepare(static_cast<int>(padded_n));
    torch::Tensor input_2d = input.contiguous().view({actual_n, k_});
    torch::Tensor output = torch::empty(out_shape, input.options());
    torch::Tensor output_2d = output.view({actual_n, m_});
    torch::Tensor b_kxn = torch::empty({k_, padded_n}, input.options());
    torch::Tensor c_mxn = torch::empty({m_, padded_n}, input.options());
    torch::Tensor d_mxn = torch::empty({m_, padded_n}, input.options());

    pack_transpose_fp16(input_2d, b_kxn, actual_n);

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    CHECK_CUSPARSE_THROW(cusparseLtMatmul(
        &handle_, &cache->plan, &alpha, cache->compressed.data_ptr(), b_kxn.data_ptr(), &beta,
        c_mxn.data_ptr(), d_mxn.data_ptr(),
        cache->workspace.defined() ? cache->workspace.data_ptr() : nullptr, &stream, 1));

    unpack_transpose_bias_fp16(d_mxn, bias_, output_2d, actual_n, has_bias_);
    return output;
  }

  double prepare_for_n(int64_t n) {
    auto start = std::chrono::high_resolution_clock::now();
    prepare(static_cast<int>(padded_n8(n)));
    CHECK_CUDA_THROW(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }

  int out_features() const { return m_; }
  int in_features() const { return k_; }

 private:
  std::shared_ptr<LtPlanCache> prepare(int n) {
    auto it = caches_.find(n);
    if (it != caches_.end()) return it->second;
    auto cache = std::make_shared<LtPlanCache>();
    cache->n = n;
    CHECK_CUSPARSE_THROW(cusparseLtStructuredDescriptorInit(
        &handle_, &cache->matA, m_, k_, k_, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE_THROW(cusparseLtDenseDescriptorInit(
        &handle_, &cache->matB, k_, n, n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE_THROW(cusparseLtDenseDescriptorInit(
        &handle_, &cache->matC, m_, n, n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE_THROW(cusparseLtMatmulDescriptorInit(
        &handle_, &cache->matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &cache->matA, &cache->matB, &cache->matC, &cache->matC, CUSPARSE_COMPUTE_32F));
    int valid = 0;
    CHECK_CUSPARSE_THROW(cusparseLtSpMMAPruneCheck(&handle_, &cache->matmul, weight_.data_ptr(), &valid, nullptr));
    if (valid != 0) {
      throw std::runtime_error("weight failed cuSPARSELt 2:4 prune check");
    }
    CHECK_CUSPARSE_THROW(cusparseLtMatmulAlgSelectionInit(&handle_, &cache->alg, &cache->matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE_THROW(cusparseLtMatmulPlanInit(&handle_, &cache->plan, &cache->matmul, &cache->alg));
    CHECK_CUSPARSE_THROW(cusparseLtSpMMACompressedSize(&handle_, &cache->plan, &cache->compressed_size, &cache->compress_buffer_size));
    CHECK_CUSPARSE_THROW(cusparseLtMatmulGetWorkspace(&handle_, &cache->plan, &cache->workspace_size));
    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(weight_.device());
    cache->compressed = torch::empty({static_cast<int64_t>(cache->compressed_size)}, byte_opts);
    if (cache->compress_buffer_size > 0) {
      cache->compress_buffer = torch::empty({static_cast<int64_t>(cache->compress_buffer_size)}, byte_opts);
    }
    if (cache->workspace_size > 0) {
      cache->workspace = torch::empty({static_cast<int64_t>(cache->workspace_size)}, byte_opts);
    }
    CHECK_CUSPARSE_THROW(cusparseLtSpMMACompress(
        &handle_, &cache->plan, weight_.data_ptr(), cache->compressed.data_ptr(),
        cache->compress_buffer.defined() ? cache->compress_buffer.data_ptr() : nullptr,
        at::cuda::getCurrentCUDAStream()));
    cache->valid = true;
    caches_[n] = cache;
    return cache;
  }

  alignas(16) cusparseLtHandle_t handle_{};
  torch::Tensor weight_;
  torch::Tensor bias_;
  bool has_bias_ = false;
  int m_ = 0;
  int k_ = 0;
  std::unordered_map<int, std::shared_ptr<LtPlanCache>> caches_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<CuSparseLtLinearCpp, std::shared_ptr<CuSparseLtLinearCpp>>(m, "CuSparseLtLinearCpp")
      .def(pybind11::init<torch::Tensor, torch::Tensor>())
      .def("forward", &CuSparseLtLinearCpp::forward)
      .def("prepare_for_n", &CuSparseLtLinearCpp::prepare_for_n)
      .def_property_readonly("in_features", &CuSparseLtLinearCpp::in_features)
      .def_property_readonly("out_features", &CuSparseLtLinearCpp::out_features);
}
