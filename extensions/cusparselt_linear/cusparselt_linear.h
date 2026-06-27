#pragma once

#include <torch/extension.h>

void pack_transpose_fp16(torch::Tensor input_2d, torch::Tensor packed_kxn, int64_t actual_n);
void unpack_transpose_bias_fp16(torch::Tensor d_mxn, torch::Tensor bias, torch::Tensor output_2d, int64_t actual_n, bool has_bias);

