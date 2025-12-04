#pragma once
#include "nn/core/tensor.h"

Tensor* relu(Tensor* input);
Tensor* relu_grad(Tensor* input, Tensor* grad_output);