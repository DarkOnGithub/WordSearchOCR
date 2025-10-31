#pragma once

#include "nn/core/tensor.h"

typedef struct {
    Tensor* weights;
    Tensor* biases;
    Tensor* weight_grad;
    Tensor* bias_grad;
} LayerGrad;

LayerGrad* layer_grad_create(Tensor* weights, Tensor* biases);
void layer_grad_free(LayerGrad* layer_grad);