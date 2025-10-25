#include "tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include <stdint.h>
#include <stddef.h>

static inline void simd_add_float(float* dest, const float* a, const float* b, size_t count) {
    size_t i = 0;
    size_t vector_count = count / 8;

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&dest[i], result);
    }

    for (; i < count; ++i) {
        dest[i] = a[i] + b[i];
    }
}

static inline void simd_sub_float(float* dest, const float* a, const float* b, size_t count) {
    size_t i = 0;
    size_t vector_count = count / 8;

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&dest[i], result);
    }

    for (; i < count; ++i) {
        dest[i] = a[i] - b[i];
    }
}

static inline void simd_mul_float(float* dest, const float* a, const float* b, size_t count) {
    size_t i = 0;
    size_t vector_count = count / 8;

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&dest[i], result);
    }

    for (; i < count; ++i) {
        dest[i] = a[i] * b[i];
    }
}

inline void simd_set_float(float* dest, float value, size_t count) {
    size_t i = 0;
    size_t vector_count = count / 8;
    __m256 val_vec = _mm256_set1_ps(value);

    for (i = 0; i < vector_count * 8; i += 8) {
        _mm256_storeu_ps(&dest[i], val_vec);
    }

    for (; i < count; ++i) {
        dest[i] = value;
    }
}

static float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

Tensor* tensor_create(int batch_size, int channels, int width, int height) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;

    tensor->batch_size = batch_size;
    tensor->channels = channels;
    tensor->width = width;
    tensor->height = height;
    tensor->size = batch_size * channels * width * height;

    tensor->data = (float*)malloc(tensor->size * sizeof(float));
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }

    return tensor;
}

Tensor* tensor_create_zero(int batch_size, int channels, int width, int height) {
    Tensor* tensor = tensor_create(batch_size, channels, width, height);
    if (!tensor) return NULL;

    simd_set_float(tensor->data, 0.0f, tensor->size);
    return tensor;
}

Tensor* tensor_create_ones(int batch_size, int channels, int width, int height) {
    Tensor* tensor = tensor_create(batch_size, channels, width, height);
    if (!tensor) return NULL;

    simd_set_float(tensor->data, 1.0f, tensor->size);
    return tensor;
}

Tensor* tensor_create_random(int batch_size, int channels, int width, int height) {
    Tensor* tensor = tensor_create(batch_size, channels, width, height);
    if (!tensor) return NULL;

    srand((unsigned int)time(NULL));

    for (int i = 0; i < tensor->size; ++i) {
        tensor->data[i] = random_float();
    }

    return tensor;
}

Tensor* tensor_from_data(float* data, int batch_size, int channels, int width, int height) {
    Tensor* tensor = tensor_create(batch_size, channels, width, height);
    if (!tensor) return NULL;

    memcpy(tensor->data, data, tensor->size * sizeof(float));
    return tensor;
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor);
    }
}

Tensor* tensor_flatten(Tensor* tensor) {
    // Result shape: (batch_size, channels * height * width)
    int flattened_size = tensor->channels * tensor->height * tensor->width;
    Tensor* flattened = tensor_create(tensor->batch_size, flattened_size, 1, 1);
    if (!flattened) return NULL;
    memcpy(flattened->data, tensor->data, tensor->size * sizeof(float));
    tensor_free(tensor);
    return flattened;
}

Tensor* tensor_reshape(Tensor* tensor, int batch_size, int channels, int height, int width) {
    // Check if the total size matches
    int new_size = batch_size * channels * height * width;
    if (new_size != tensor->size) {
        printf("Error: Cannot reshape tensor of size %d to shape %d x %d x %d x %d (size %d)\n",
               tensor->size, batch_size, channels, height, width, new_size);
        return NULL;
    }

    Tensor* reshaped = tensor_create(batch_size, channels, height, width);
    if (!reshaped) return NULL;

    memcpy(reshaped->data, tensor->data, tensor->size * sizeof(float));
    tensor_free(tensor);
    return reshaped;
}

//!TODO: implement without keep_dim (if necessary)
/**
@param a: The tensor to sum along the specified axis
@param axis: The axis to sum along
@param keep_dim: Whether to keep the dimensions of the input tensor
@return: The summed tensor
@note: Keep_dim has to be true
*/
Tensor* tensor_sum(Tensor* a, int axis, bool keep_dim) {
    if(keep_dim == false){
        fprintf(stderr, "Error: keep_dim must be true\n");
        return NULL;
    }

    if (!a) return NULL;

    if (axis < 0 || axis > 3) {
        printf("Error: Invalid axis %d for tensor sum. Axis must be 0-3\n", axis);
        return NULL;
    }

    int out_batch_size = a->batch_size;
    int out_channels = a->channels;
    int out_height = a->height;
    int out_width = a->width;

    switch (axis) {
        case 0: out_batch_size = 1; break;
        case 1: out_channels = 1; break;
        case 2: out_height = 1; break;
        case 3: out_width = 1; break;
    }

    Tensor* result = tensor_create_zero(out_batch_size, out_channels, out_height, out_width);
    if (!result) return NULL;

    for (int b = 0; b < a->batch_size; ++b) {
        for (int c = 0; c < a->channels; ++c) {
            for (int h = 0; h < a->height; ++h) {
                for (int w = 0; w < a->width; ++w) {
                    int in_idx = b * (a->channels * a->height * a->width) +
                                c * (a->height * a->width) +
                                h * a->width + w;

                    int out_b = (axis == 0) ? 0 : b;
                    int out_c = (axis == 1) ? 0 : c;
                    int out_h = (axis == 2) ? 0 : h;
                    int out_w = (axis == 3) ? 0 : w;

                    int out_idx = out_b * (out_channels * out_height * out_width) +
                                 out_c * (out_height * out_width) +
                                 out_h * out_width + out_w;

                    result->data[out_idx] += a->data[in_idx];
                }
            }
        }
    }

    return result;
}

//Not optimized for performance, will only be used for the small nn, the CNN shouldn't use it
//Otherwise we should optimize it later
Tensor* tensor_dot_product(Tensor* a, Tensor* b, int axis) {
    if (!a || !b) return NULL;

    if (axis < 0 || axis > 3) {
        printf("Error: Invalid axis %d for tensor dot product. Axis must be 0-3\n", axis);
        return NULL;
    }

    int a_dim_size, b_dim_size;
    switch (axis) {
        case 0: a_dim_size = a->batch_size; b_dim_size = b->batch_size; break;
        case 1: a_dim_size = a->channels; b_dim_size = b->channels; break;
        case 2: a_dim_size = a->height; b_dim_size = b->height; break;
        case 3: a_dim_size = a->width; b_dim_size = b->width; break;
    }

    if (a_dim_size != b_dim_size) {
        printf("Error: Tensor dimensions don't match for dot product along axis %d: %d vs %d\n",
               axis, a_dim_size, b_dim_size);
        return NULL;
    }

    if ((axis != 0 && a->batch_size != b->batch_size) ||
        (axis != 1 && a->channels != b->channels) ||
        (axis != 2 && a->height != b->height) ||
        (axis != 3 && a->width != b->width)) {
        printf("Error: Tensor shapes don't match for dot product along axis %d\n", axis);
        printf("Tensor A shape: [%d, %d, %d, %d]\n", a->batch_size, a->channels, a->height, a->width);
        printf("Tensor B shape: [%d, %d, %d, %d]\n", b->batch_size, b->channels, b->height, b->width);
        return NULL;
    }

    int out_batch_size = a->batch_size;
    int out_channels = a->channels;
    int out_height = a->height;
    int out_width = a->width;

    switch (axis) {
        case 0: out_batch_size = 1; break;
        case 1: out_channels = 1; break;
        case 2: out_height = 1; break;
        case 3: out_width = 1; break;
    }

    Tensor* result = tensor_create_zero(out_batch_size, out_channels, out_height, out_width);
    if (!result) return NULL;

    for (int i_b = 0; i_b < a->batch_size; ++i_b) {
        for (int i_c = 0; i_c < a->channels; ++i_c) {
            for (int i_h = 0; i_h < a->height; ++i_h) {
                for (int i_w = 0; i_w < a->width; ++i_w) {
                    if ((axis == 0 && i_b != 0) ||
                        (axis == 1 && i_c != 0) ||
                        (axis == 2 && i_h != 0) ||
                        (axis == 3 && i_w != 0)) {
                        continue;
                    }

                    float sum = 0.0f;
                    int dim_size = (axis == 0) ? a->batch_size :
                                   (axis == 1) ? a->channels :
                                   (axis == 2) ? a->height : a->width;

                    for (int k = 0; k < dim_size; ++k) {
                        int a_b = (axis == 0) ? k : i_b;
                        int a_c = (axis == 1) ? k : i_c;
                        int a_h = (axis == 2) ? k : i_h;
                        int a_w = (axis == 3) ? k : i_w;

                        int b_b = (axis == 0) ? k : i_b;
                        int b_c = (axis == 1) ? k : i_c;
                        int b_h = (axis == 2) ? k : i_h;
                        int b_w = (axis == 3) ? k : i_w;

                        int a_idx = a_b * (a->channels * a->height * a->width) +
                                    a_c * (a->height * a->width) +
                                    a_h * a->width + a_w;

                        int b_idx = b_b * (b->channels * b->height * b->width) +
                                    b_c * (b->height * b->width) +
                                    b_h * b->width + b_w;

                        sum += a->data[a_idx] * b->data[b_idx];
                    }

                    int out_b = (axis == 0) ? 0 : i_b;
                    int out_c = (axis == 1) ? 0 : i_c;
                    int out_h = (axis == 2) ? 0 : i_h;
                    int out_w = (axis == 3) ? 0 : i_w;

                    int out_idx = out_b * (out_channels * out_height * out_width) +
                                  out_c * (out_height * out_width) +
                                  out_h * out_width + out_w;

                    result->data[out_idx] = sum;
                }
            }
        }
    }

    return result;
}


Tensor* tensor_matmul(Tensor* a, Tensor* b, Tensor* bias) {
    if (!a || !b) return NULL;

    if (a->height != 1 || a->width != 1) {
        printf("Error: Input tensor A must have height=1, width=1 for matmul\n");
        return NULL;
    }
    if (b->height != 1 || b->width != 1) {
        printf("Error: Weight tensor B must have height=1, width=1 for matmul\n");
        return NULL;
    }
    if (bias && (bias->batch_size != 1 || bias->height != 1 || bias->width != 1)) {
        printf("Error: Bias tensor must have batch_size=1, height=1, width=1 for matmul\n");
        return NULL;
    }

    int batch_size = a->batch_size;
    int input_size = a->channels;
    int output_size = b->batch_size;

    if (b->channels != input_size) {
        printf("Error: Input channels (%d) must match weight channels (%d)\n", input_size, b->channels);
        return NULL;
    }
    if (bias && bias->channels != output_size) {
        printf("Error: Bias channels (%d) must match output size (%d)\n", bias->channels, output_size);
        return NULL;
    }

    Tensor* result = tensor_create(batch_size, output_size, 1, 1);
    if (!result) return NULL;

    for (int batch = 0; batch < batch_size; ++batch) {
        const float* A_row = &a->data[batch * input_size];
        for (int out = 0; out < output_size; ++out) {
            const float* B_row = &b->data[out * input_size];
            __m256 sum_vec = _mm256_setzero_ps();
            float sum_scalar = bias ? bias->data[out] : 0.0f;

            size_t i = 0;
            size_t vector_count = input_size / 8;

            for (i = 0; i < vector_count * 8; i += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A_row[i]);
                __m256 b_vec = _mm256_loadu_ps(&B_row[i]);
                __m256 prod_vec = _mm256_mul_ps(a_vec, b_vec);
                sum_vec = _mm256_add_ps(sum_vec, prod_vec);
            }

            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            for (int j = 0; j < 8; ++j) {
                sum_scalar += temp[j];
            }

            for (; i < input_size; ++i) {
                sum_scalar += A_row[i] * B_row[i];
            }

            result->data[batch * output_size + out] = sum_scalar;
        }
    }

    return result;
}

void tensor_print(Tensor* tensor) {
    if (!tensor) {
        printf("NULL tensor\n");
        return;
    }

    printf("tensor(");

    const int max_display_width = 6;
    const int max_display_height = 4;
    const int max_display_batch = 2;

    if (tensor->height == 1 && tensor->batch_size == 1 && tensor->channels == 1) {
        printf("[");
        for (int w = 0; w < tensor->width; ++w) {
            if (tensor->width > max_display_width && w == max_display_width / 2) {
                printf("..., ");
                w = tensor->width - max_display_width / 2 - 1;
                continue;
            }
            printf("%.4f", tensor->data[w]);
            if (w < tensor->width - 1) printf(", ");
        }
        printf("]");
    } else {
        printf("[");

        for (int b = 0; b < tensor->batch_size; ++b) {
            if (b > 0) printf(",\n      ");

            if (tensor->batch_size > max_display_batch && b == max_display_batch / 2) {
                printf("...");
                b = tensor->batch_size - max_display_batch / 2 - 1;
                continue;
            }

            if (tensor->batch_size > max_display_batch && b >= max_display_batch / 2) {
                // Continue printing batches
            }

            printf("[");

            for (int c = 0; c < tensor->channels; ++c) {
                if (c > 0) printf(",\n       ");

                printf("[");

                for (int h = 0; h < tensor->height; ++h) {
                    if (h > 0) printf(",\n        ");

                    if (tensor->height > max_display_height && h == max_display_height / 2) {
                        printf(" ...");
                        h = tensor->height - max_display_height / 2 - 1;
                        continue;
                    }

                    printf("[");

                    for (int w = 0; w < tensor->width; ++w) {
                        if (tensor->width > max_display_width && w == max_display_width / 2) {
                            printf("..., ");
                            w = tensor->width - max_display_width / 2 - 1;
                            continue;
                        }

                        int idx = b * (tensor->channels * tensor->width * tensor->height) +
                                 c * (tensor->width * tensor->height) +
                                 h * tensor->width + w;
                        printf("%.4f", tensor->data[idx]);
                        if (w < tensor->width - 1) printf(", ");
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("]");
        }
        printf("]");
    }

    printf(", shape=torch.Size([%d, %d, %d, %d]))\n",
           tensor->batch_size, tensor->channels, tensor->height, tensor->width);
}

Tensor* tensor_add(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->batch_size != b->batch_size ||
        a->channels != b->channels ||
        a->width != b->width ||
        a->height != b->height) {
        printf("Error: Tensor dimensions don't match for addition\n");
        return NULL;
    }

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    simd_add_float(result->data, a->data, b->data, a->size);
    return result;
}

Tensor* tensor_sub(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->batch_size != b->batch_size ||
        a->channels != b->channels ||
        a->width != b->width ||
        a->height != b->height) {
        printf("Error: Tensor dimensions don't match for subtraction\n");
        return NULL;
    }

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    simd_sub_float(result->data, a->data, b->data, a->size);
    return result;
}

Tensor* tensor_mul(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->batch_size != b->batch_size ||
        a->channels != b->channels ||
        a->width != b->width ||
        a->height != b->height) {
        printf("Error: Tensor dimensions don't match for multiplication\n");
        return NULL;
    }

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    simd_mul_float(result->data, a->data, b->data, a->size);
    return result;
}

Tensor* tensor_square(Tensor* a) {
    if (!a) return NULL;

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * a->data[i];
    }
    return result;
}

Tensor* tensor_sqrt(Tensor* a) {
    if (!a) return NULL;

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = sqrtf(a->data[i]);
    }
    return result;
}

Tensor* tensor_exp(Tensor* a) {
    if(!a)return NULL;

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if(!result)return NULL;
    for(size_t i = 0; i < a->size; i++){
        result->data[i] = exp(a->data[i]);
    }
    return result;
}

Tensor* tensor_div(Tensor* a, Tensor* b) {
    if (!a || !b) return NULL;

    if (a->batch_size != b->batch_size ||
        a->channels != b->channels ||
        a->width != b->width ||
        a->height != b->height) {
        printf("Error: Tensor dimensions don't match for division\n");
        return NULL;
    }

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] / b->data[i];
    }
    return result;
}

Tensor* tensor_mul_scalar(Tensor* a, float scalar) {
    if (!a) return NULL;

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * scalar;
    }
    return result;
}

Tensor* tensor_add_scalar(Tensor* a, float scalar) {
    if (!a) return NULL;

    Tensor* result = tensor_create(a->batch_size, a->channels, a->width, a->height);
    if (!result) return NULL;

    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + scalar;
    }
    return result;
}

void tensor_scale_inplace(Tensor* a, float scalar) {
    if (!a) return;

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] *= scalar;
    }
}

void tensor_add_scalar_inplace(Tensor* a, float scalar) {
    if (!a) return;

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] += scalar;
    }
}