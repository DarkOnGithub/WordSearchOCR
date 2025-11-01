#pragma once

#include "nn/core/tensor.h"

typedef struct {
    Tensor* weight;       // [output_channels, input_channels, kernel_size, kernel_size]
    Tensor* bias;         // [output_channels, 1, 1, 1]
    Tensor* weight_grad;  // Same shape as weight
    Tensor* bias_grad;    // Same shape as bias
    int kernel_size;
    int stride;
    int padding;
} Conv2D;

Conv2D* conv2D_create(int input_channels, int output_channels, int kernel_size, int stride, int padding);
void conv2D_free(Conv2D* conv2D);
Tensor* conv2D_forward(Conv2D* conv2D, Tensor* input);
Tensor* conv2D_backward(Conv2D* conv2D, Tensor* input, Tensor* grad_output);
void conv2D_zero_grad(Conv2D* conv2D);