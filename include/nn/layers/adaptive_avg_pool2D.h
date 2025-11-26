#pragma once
#include "nn/core/tensor.h"

typedef struct {
    int output_height;
    int output_width;
    // Cache for backward pass
    Tensor* input_cache;
} AdaptiveAvgPool2D;

typedef struct {
    Tensor* output;
    AdaptiveAvgPool2D* layer;
} AdaptiveAvgPool2DOutput;

typedef struct {
    Tensor* input_grad;
} AdaptiveAvgPool2DBackwardOutput;

AdaptiveAvgPool2D* adaptive_avg_pool2d_create(int output_height, int output_width);
AdaptiveAvgPool2DOutput* adaptive_avg_pool2d_forward(AdaptiveAvgPool2D* layer, Tensor* input);
AdaptiveAvgPool2DBackwardOutput* adaptive_avg_pool2d_backward(AdaptiveAvgPool2D* layer, AdaptiveAvgPool2DOutput* forward_result, Tensor* output_grad);

void adaptive_avg_pool2d_free(AdaptiveAvgPool2D* layer);
void adaptive_avg_pool2d_output_free(AdaptiveAvgPool2DOutput* result);
void adaptive_avg_pool2d_backward_output_free(AdaptiveAvgPool2DBackwardOutput* result);
