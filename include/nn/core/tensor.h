#pragma once

typedef struct {
    float *data;
    int* shape;
    int size;
    int ndim;
} Tensor;

Tensor* tensor_create(int* shape, int ndim);
Tensor* tensor_create_zero(int* shape, int ndim);
Tensor* tensor_create_ones(int* shape, int ndim);
Tensor* tensor_create_random(int* shape, int ndim);
Tensor* tensor_from_data(float* data, int* shape, int ndim);
void tensor_free(Tensor* tensor);
void tensor_print(Tensor* tensor);

