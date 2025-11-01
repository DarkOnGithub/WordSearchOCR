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

Tensor* tensor_add(Tensor* tensor1, Tensor* tensor2);
void tensor_add_inplace(Tensor* tensor1, Tensor* tensor2);
void tensor_add_scalar(Tensor* tensor, float scalar);
Tensor* tensor_add_scalar_copy(Tensor* tensor, float scalar);
Tensor* tensor_subtract(Tensor* tensor1, Tensor* tensor2);
void tensor_subtract_inplace(Tensor* tensor1, Tensor* tensor2);
void tensor_subtract_scalar(Tensor* tensor, float scalar);
Tensor* tensor_multiply(Tensor* tensor1, Tensor* tensor2);
void tensor_multiply_inplace(Tensor* tensor1, Tensor* tensor2);
void tensor_multiply_scalar(Tensor* tensor, float scalar);
Tensor* tensor_multiply_scalar_copy(Tensor* tensor, float scalar);
Tensor* tensor_divide(Tensor* tensor1, Tensor* tensor2);
Tensor* tensor_square(Tensor* tensor);
Tensor* tensor_sqrt(Tensor* tensor);
Tensor* tensor_power(Tensor* tensor, float power);
void tensor_power_inplace(Tensor* tensor, float power);
void tensor_add_bias_inplace(Tensor* tensor, const Tensor* bias);
Tensor* tensor_sum_axis(const Tensor* tensor, int axis);
void tensor_outer_product_accumulate(Tensor* result, const Tensor* a, const Tensor* b);
Tensor* tensor_transpose(const Tensor* tensor);
void tensor_scale_inplace(Tensor* tensor, float scale);