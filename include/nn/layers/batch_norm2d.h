#pragma once
#include "nn/core/tensor.h"
#include "../core/layer_grad.h"

typedef struct {
    int num_features;  // number of channels
    float momentum;
    float epsilon;
    int training;
    LayerGrad* layer_grad;  // gamma and beta parameters
    Tensor* running_mean;
    Tensor* running_var;
    // Cache for backward pass
    Tensor* input_cache;
    Tensor* normalized_cache;
    Tensor* std_cache;
    Tensor* var_cache;
    Tensor* mean_cache;
} BatchNorm2D;

typedef struct {
    Tensor* output;
    BatchNorm2D* layer;
} BatchNorm2DOutput;

typedef struct {
    Tensor* input_grad;
} BatchNorm2DBackwardOutput;

BatchNorm2D* batch_norm2d_create(int num_features, float momentum, float epsilon);
BatchNorm2DOutput* batch_norm2d_forward(BatchNorm2D* layer, Tensor* input);
BatchNorm2DBackwardOutput* batch_norm2d_backward(BatchNorm2D* layer, BatchNorm2DOutput* forward_result, Tensor* output_grad);

void batch_norm2d_free(BatchNorm2D* layer);
void batch_norm2d_output_free(BatchNorm2DOutput* result);
void batch_norm2d_backward_output_free(BatchNorm2DBackwardOutput* result);
void batch_norm2d_set_training(BatchNorm2D* layer, int training);
