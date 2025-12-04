#pragma once
#include <stdbool.h>
#include "nn/core/tensor.h"

typedef struct {
    float dropout_rate;
    bool training;
} Dropout2D;

typedef struct {
    Tensor* output;
    Tensor* mask;
} Dropout2DOutput;

typedef struct {
    Tensor* input_grad;
} Dropout2DBackwardOutput;

Dropout2D* dropout2d_create(float dropout_rate);
void dropout2d_free(Dropout2D* layer);
Dropout2DOutput* dropout2d_forward(Dropout2D* layer, Tensor* input);
void dropout2d_output_free(Dropout2DOutput* result);
Dropout2DBackwardOutput* dropout2d_backward(Dropout2D* layer, Dropout2DOutput* forward_result,
                                           Tensor* output_grad);
void dropout2d_backward_output_free(Dropout2DBackwardOutput* result);
