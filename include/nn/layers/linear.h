#pragma once
#include "nn/core/tensor.h"
#include "../core/layer_grad.h"

typedef struct {
    int input_size;
    int output_size;
    LayerGrad* layer_grad;
    Tensor* input_cache;
} Linear;

typedef struct {
    Tensor* output;
    Linear* layer;
} LinearOutput;

typedef struct {
    Tensor* input_grad;
} LinearBackwardOutput;

Linear* linear_create(int input_size, int output_size);
LinearOutput* linear_forward(Linear* layer, Tensor* input);
LinearBackwardOutput* linear_backward(Linear* layer, LinearOutput* forward_result, Tensor* output_grad);

void linear_free(Linear* layer);
void linear_output_free(LinearOutput* result);
void linear_backward_output_free(LinearBackwardOutput* result);