#pragma once
#include "nn/core/tensor.h"

Tensor* silu(Tensor* input);
Tensor* silu_grad(Tensor* input, Tensor* grad_output);