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

void tensor_print(Tensor* tensor) {
    if (!tensor) {
        printf("NULL tensor\n");
        return;
    }

    printf("tensor(");

    const int max_display_width = 6;
    const int max_display_height = 4;
    const int max_display_batch = 2;

    // Handle 1D tensors (flattened or single batch with height=1, channels=1)
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