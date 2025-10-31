#include "nn/core/tensor.h"
#include "nn/core/utils.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

Tensor* tensor_create(int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    tensor->shape = shape;
    tensor->ndim = ndim;
    tensor->size = 1;
    tensor->data = (float*)malloc(sizeof(float) * tensor->size);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    return tensor;
}

Tensor* tensor_create_zero(int* shape, int ndim) {
    Tensor* tensor = tensor_create(shape, ndim);
    if (!tensor) return NULL;
    memset(tensor->data, 0, sizeof(float) * tensor->size);
    return tensor;
}

Tensor* tensor_create_ones(int* shape, int ndim) {
    Tensor* tensor = tensor_create(shape, ndim);
    if (!tensor) return NULL;
    memset(tensor->data, 1, sizeof(float) * tensor->size);
    return tensor;
}

Tensor* tensor_create_random(int* shape, int ndim) {
    Tensor* tensor = tensor_create(shape, ndim);
    if (!tensor) return NULL;
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = random_float();
    }
    return tensor;
}

Tensor* tensor_from_data(float* data, int* shape, int ndim) {
    Tensor* tensor = tensor_create(shape, ndim);
    if (!tensor) return NULL;
    memcpy(tensor->data, data, sizeof(float) * tensor->size);
    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
    }
}
