#include <stddef.h>
#include <stdbool.h>
#pragma once

typedef struct {
    int size;
    int batch_size;
    int channels;
    int width;
    int height;
    float* data;
} Tensor;

Tensor* tensor_create(int batch_size, int channels, int width, int height);
Tensor* tensor_create_zero(int batch_size, int channels, int width, int height);
Tensor* tensor_create_ones(int batch_size, int channels, int width, int height);
Tensor* tensor_create_random(int batch_size, int channels, int width, int height);
Tensor* tensor_from_data(float* data, int batch_size, int channels, int width, int height);
void tensor_free(Tensor* tensor);

//utils
void tensor_print(Tensor* tensor);
Tensor* tensor_flatten(Tensor* tensor);
Tensor* tensor_reshape(Tensor* tensor, int batch_size, int channels, int height, int width);
Tensor* tensor_sum(Tensor* a, int axis, bool keep_dim);
Tensor* tensor_dot_product(Tensor* a, Tensor* b, int axis);
Tensor* tensor_matmul(Tensor* a, Tensor* b, Tensor* bias);

//Operations
Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_mul(Tensor* a, Tensor* b);


// Element-wise operations
Tensor* tensor_square(Tensor* a);
Tensor* tensor_sqrt(Tensor* a);
Tensor* tensor_div(Tensor* a, Tensor* b);
Tensor* tensor_exp(Tensor* a);


// Scalar operations
Tensor* tensor_mul_scalar(Tensor* a, float scalar);
Tensor* tensor_add_scalar(Tensor* a, float scalar);
Tensor* tensor_div_scalar(Tensor* a, float scalar);

// In-place operations
void tensor_scale_inplace(Tensor* a, float scalar);
void tensor_add_scalar_inplace(Tensor* a, float scalar);

void simd_set_float(float* dest, float value, size_t count);