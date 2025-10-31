#include "nn/core/layer_grad.h"
#include <stdlib.h>
#include <string.h>

LayerGrad* layer_grad_create(Tensor* weights, Tensor* biases) {
    LayerGrad* layer_grad = (LayerGrad*)malloc(sizeof(LayerGrad));
    if (!layer_grad) return NULL;
    layer_grad->weights = weights;
    layer_grad->biases = biases;
    layer_grad->weight_grad = tensor_create_zero(weights->shape, weights->ndim);
    layer_grad->bias_grad = tensor_create_zero(biases->shape, biases->ndim);
    return layer_grad;
}

void layer_zero_grad(LayerGrad* layer_grad) {
    if (!layer_grad) return;
    memset(layer_grad->weight_grad->data, 0, sizeof(float) * layer_grad->weight_grad->size);
    memset(layer_grad->bias_grad->data, 0, sizeof(float) * layer_grad->bias_grad->size);
}

void layer_grad_free(LayerGrad* layer_grad) {
    if (!layer_grad) return;
    tensor_free(layer_grad->weights);
    tensor_free(layer_grad->biases);
    tensor_free(layer_grad->weight_grad);
    tensor_free(layer_grad->bias_grad);
    free(layer_grad);
}