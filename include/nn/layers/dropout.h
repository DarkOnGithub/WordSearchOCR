#pragma once
#include <stdbool.h>
#include "nn/core/tensor.h"

typedef struct {
    float dropout_rate;
    bool training;
} Dropout;

typedef struct {
    Tensor* output;
    Tensor* mask;
} DropoutOutput;

typedef struct {
    Tensor* input_grad;
} DropoutBackwardOutput;

Dropout* dropout_create(float dropout_rate);
void dropout_free(Dropout* layer);
DropoutOutput* dropout_forward(Dropout* layer, Tensor* input);
void dropout_output_free(DropoutOutput* result);

DropoutBackwardOutput* dropout_backward(Dropout* layer, DropoutOutput* forward_result,
                                       Tensor* output_grad);
void dropout_backward_output_free(DropoutBackwardOutput* result);
