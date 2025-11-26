#include "nn/core/tensor.h"
#include "nn/core/utils.h"
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

Tensor* tensor_create(int* shape, int ndim) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (!tensor) return NULL;
    tensor->shape = (int*)malloc(sizeof(int) * ndim);
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, sizeof(int) * ndim);
    tensor->ndim = ndim;
    tensor->size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->size *= shape[i];
    }
    tensor->data = (float*)malloc(sizeof(float) * tensor->size);
    if (!tensor->data) {
        free(tensor->shape);
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
    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = 1.0f;
    }
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



static void tensor_print_recursive(const Tensor* tensor, int* strides, int* indices, int current_dim, int offset) {
    if (current_dim == tensor->ndim) {
        printf("%.4f", tensor->data[offset]);
        return;
    }

    int dim_size = tensor->shape[current_dim];

    printf("[");

    for (int i = 0; i < dim_size; ++i) {
        if (i > 0) {
            if (current_dim == 0) printf(",\n ");
            else printf(", ");
        }

        indices[current_dim] = i;
        int next_offset = offset + i * strides[current_dim];
        tensor_print_recursive(tensor, strides, indices, current_dim + 1, next_offset);
    }

    printf("]");
}

void tensor_print(Tensor* tensor) {
    if (!tensor) {
        printf("NULL tensor\n");
        return;
    }

    if (tensor->ndim == 0) {
        printf("tensor(%.4f)\n", tensor->data[0]);
        return;
    }

    printf("tensor(");

    int* strides = (int*)malloc(sizeof(int) * tensor->ndim);
    if (!strides) {
        printf("Failed to allocate memory for strides\n");
        return;
    }

    strides[tensor->ndim - 1] = 1;
    for (int i = tensor->ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * tensor->shape[i + 1];
    }

    int* indices = (int*)malloc(sizeof(int) * tensor->ndim);
    if (!indices) {
        free(strides);
        printf("Failed to allocate memory for indices\n");
        return;
    }

    tensor_print_recursive(tensor, strides, indices, 0, 0);

    printf(", shape=Size([");
    for (int i = 0; i < tensor->ndim; ++i) {
        if (i > 0) printf(", ");
        printf("%d", tensor->shape[i]);
    }
    printf("]))\n");

    free(indices);
    free(strides);
}

static int tensors_compatible(const Tensor* t1, const Tensor* t2) {
    if (t1->ndim != t2->ndim) return 0;
    for (int i = 0; i < t1->ndim; i++) {
        if (t1->shape[i] != t2->shape[i]) return 0;
    }
    return 1;
}

Tensor* tensor_add(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return NULL;
    }

    Tensor* result = tensor_create(tensor1->shape, tensor1->ndim);
    if (!result) return NULL;

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 sum = _mm256_add_ps(vec1, vec2);
        _mm256_storeu_ps(&result_data[i], sum);
    }

    for (; i < size; i++) {
        result_data[i] = data1[i] + data2[i];
    }

    return result;
}

void tensor_add_inplace(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return;
    }

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 sum = _mm256_add_ps(vec1, vec2);
        _mm256_storeu_ps(&data1[i], sum);
    }

    for (; i < size; i++) {
        data1[i] += data2[i];
    }
}

void tensor_add_scalar(Tensor* tensor, float scalar) {
    if (!tensor) return;

    int size = tensor->size;
    float* data = tensor->data;

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        __m256 sum = _mm256_add_ps(vec, scalar_vec);
        _mm256_storeu_ps(&data[i], sum);
    }

    for (; i < size; i++) {
        data[i] += scalar;
    }
}

Tensor* tensor_add_scalar_copy(Tensor* tensor, float scalar) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->shape, tensor->ndim);
    if (!result) return NULL;

    memcpy(result->data, tensor->data, tensor->size * sizeof(float));

    tensor_add_scalar(result, scalar);

    return result;
}

Tensor* tensor_subtract(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return NULL;
    }

    Tensor* result = tensor_create(tensor1->shape, tensor1->ndim);
    if (!result) return NULL;

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 diff = _mm256_sub_ps(vec1, vec2);
        _mm256_storeu_ps(&result_data[i], diff);
    }

    for (; i < size; i++) {
        result_data[i] = data1[i] - data2[i];
    }

    return result;
}

void tensor_subtract_inplace(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return;
    }

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 diff = _mm256_sub_ps(vec1, vec2);
        _mm256_storeu_ps(&data1[i], diff);
    }

    for (; i < size; i++) {
        data1[i] -= data2[i];
    }
}

void tensor_subtract_scalar(Tensor* tensor, float scalar) {
    if (!tensor) return;

    int size = tensor->size;
    float* data = tensor->data;

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        __m256 diff = _mm256_sub_ps(vec, scalar_vec);
        _mm256_storeu_ps(&data[i], diff);
    }

    for (; i < size; i++) {
        data[i] -= scalar;
    }
}

Tensor* tensor_multiply(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return NULL;
    }

    Tensor* result = tensor_create(tensor1->shape, tensor1->ndim);
    if (!result) return NULL;

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 prod = _mm256_mul_ps(vec1, vec2);
        _mm256_storeu_ps(&result_data[i], prod);
    }

    for (; i < size; i++) {
        result_data[i] = data1[i] * data2[i];
    }

    return result;
}

void tensor_multiply_inplace(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return;
    }

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 prod = _mm256_mul_ps(vec1, vec2);
        _mm256_storeu_ps(&data1[i], prod);
    }

    for (; i < size; i++) {
        data1[i] *= data2[i];
    }
}

void tensor_multiply_scalar(Tensor* tensor, float scalar) {
    if (!tensor) return;

    int size = tensor->size;
    float* data = tensor->data;

    __m256 scalar_vec = _mm256_set1_ps(scalar);

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        __m256 prod = _mm256_mul_ps(vec, scalar_vec);
        _mm256_storeu_ps(&data[i], prod);
    }

    for (; i < size; i++) {
        data[i] *= scalar;
    }
}

Tensor* tensor_multiply_scalar_copy(Tensor* tensor, float scalar) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->shape, tensor->ndim);
    if (!result) return NULL;

    memcpy(result->data, tensor->data, tensor->size * sizeof(float));

    tensor_multiply_scalar(result, scalar);

    return result;
}

Tensor* tensor_divide(Tensor* tensor1, Tensor* tensor2) {
    if (!tensor1 || !tensor2 || !tensors_compatible(tensor1, tensor2)) {
        return NULL;
    }

    Tensor* result = tensor_create(tensor1->shape, tensor1->ndim);
    if (!result) return NULL;

    int size = tensor1->size;
    float* data1 = tensor1->data;
    float* data2 = tensor2->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec1 = _mm256_loadu_ps(&data1[i]);
        __m256 vec2 = _mm256_loadu_ps(&data2[i]);
        __m256 quotient = _mm256_div_ps(vec1, vec2);
        _mm256_storeu_ps(&result_data[i], quotient);
    }

    for (; i < size; i++) {
        result_data[i] = data1[i] / data2[i];
    }

    return result;
}

Tensor* tensor_square(Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->shape, tensor->ndim);
    if (!result) return NULL;

    int size = tensor->size;
    float* data = tensor->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        __m256 squared = _mm256_mul_ps(vec, vec);
        _mm256_storeu_ps(&result_data[i], squared);
    }

    for (; i < size; i++) {
        result_data[i] = data[i] * data[i];
    }

    return result;
}

Tensor* tensor_sqrt(Tensor* tensor) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->shape, tensor->ndim);
    if (!result) return NULL;

    int size = tensor->size;
    float* data = tensor->data;
    float* result_data = result->data;

    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        __m256 sqrt_vec = _mm256_sqrt_ps(vec);
        _mm256_storeu_ps(&result_data[i], sqrt_vec);
    }

    for (; i < size; i++) {
        result_data[i] = sqrtf(data[i]);
    }

    return result;
}

Tensor* tensor_power(Tensor* tensor, float power) {
    if (!tensor) return NULL;

    Tensor* result = tensor_create(tensor->shape, tensor->ndim);
    if (!result) return NULL;

    int size = tensor->size;
    float* data = tensor->data;
    float* result_data = result->data;

    if (power == 2.0f) {
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            __m256 squared = _mm256_mul_ps(vec, vec);
            _mm256_storeu_ps(&result_data[i], squared);
        }
        for (; i < size; i++) {
            result_data[i] = data[i] * data[i];
        }
    } else if (power == 0.5f) {
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            __m256 sqrt_vec = _mm256_sqrt_ps(vec);
            _mm256_storeu_ps(&result_data[i], sqrt_vec);
        }
        for (; i < size; i++) {
            result_data[i] = sqrtf(data[i]);
        }
    } else {
        for (int i = 0; i < size; i++) {
            result_data[i] = powf(data[i], power);
        }
    }

    return result;
}

void tensor_power_inplace(Tensor* tensor, float power) {
    if (!tensor) return;

    int size = tensor->size;
    float* data = tensor->data;

    if (power == 2.0f) {
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            __m256 squared = _mm256_mul_ps(vec, vec);
            _mm256_storeu_ps(&data[i], squared);
        }
        for (; i < size; i++) {
            data[i] = data[i] * data[i];
        }
    } else if (power == 0.5f) {
        int i = 0;
        for (; i <= size - 8; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            __m256 sqrt_vec = _mm256_sqrt_ps(vec);
            _mm256_storeu_ps(&data[i], sqrt_vec);
        }
        for (; i < size; i++) {
            data[i] = sqrtf(data[i]);
        }
    } else {
        for (int i = 0; i < size; i++) {
            data[i] = powf(data[i], power);
        }
    }
}

static int can_broadcast(int* shape_a, int ndim_a, int* shape_b, int ndim_b, int** out_shape, int* out_ndim) {
    int max_ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    *out_shape = (int*)malloc(sizeof(int) * max_ndim);
    if (!*out_shape) return 0;

    *out_ndim = max_ndim;

    int* padded_a = (int*)malloc(sizeof(int) * max_ndim);
    int* padded_b = (int*)malloc(sizeof(int) * max_ndim);
    if (!padded_a || !padded_b) {
        free(*out_shape);
        free(padded_a);
        free(padded_b);
        return 0;
    }

    for (int i = 0; i < max_ndim; i++) {
        padded_a[i] = 1;
        padded_b[i] = 1;
    }

    for (int i = 0; i < ndim_a; i++) {
        padded_a[max_ndim - ndim_a + i] = shape_a[i];
    }
    for (int i = 0; i < ndim_b; i++) {
        padded_b[max_ndim - ndim_b + i] = shape_b[i];
    }

    for (int i = 0; i < max_ndim; i++) {
        if (padded_a[i] == padded_b[i]) {
            (*out_shape)[i] = padded_a[i];
        } else if (padded_a[i] == 1) {
            (*out_shape)[i] = padded_b[i];
        } else if (padded_b[i] == 1) {
            (*out_shape)[i] = padded_a[i];
        } else {
            free(*out_shape);
            free(padded_a);
            free(padded_b);
            return 0;
        }
    }

    free(padded_a);
    free(padded_b);
    return 1;
}

static void compute_strides(const int* shape, int ndim, int* strides) {
    if (ndim == 0) return;
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

static int get_linear_index(const int* indices, const int* strides, int ndim) {
    int index = 0;
    for (int i = 0; i < ndim; i++) {
        index += indices[i] * strides[i];
    }
    return index;
}

static inline float _mm256_reduce_add_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    vlow  = _mm_hadd_ps(vlow, vlow);
    vlow  = _mm_hadd_ps(vlow, vlow);
    return _mm_cvtss_f32(vlow);
}


static void tensor_sgemm_simd(int m, int n, int k, float alpha, const float* A, int lda,
                              const float* B, int ldb, float beta, float* C, int ldc) {
    if (beta == 0.0f) {
        memset(C, 0, sizeof(float) * m * ldc);
    } else if (beta != 1.0f) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    if (alpha == 0.0f) return;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            const float* a_ptr = &A[i * lda];
            const float* b_ptr = &B[j];

            int l = 0;
            __m256 sum_vec = _mm256_setzero_ps();

            for (; l <= k - 8; l += 8) {
                __m256 a_vec = _mm256_loadu_ps(&a_ptr[l]);
                __m256 b_vec = _mm256_setr_ps(
                    b_ptr[(l+0) * ldb], b_ptr[(l+1) * ldb], b_ptr[(l+2) * ldb], b_ptr[(l+3) * ldb],
                    b_ptr[(l+4) * ldb], b_ptr[(l+5) * ldb], b_ptr[(l+6) * ldb], b_ptr[(l+7) * ldb]
                );
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            sum += _mm256_reduce_add_ps(sum_vec);

            for (; l < k; l++) {
                sum += a_ptr[l] * b_ptr[l * ldb];
            }

            C[i * ldc + j] += alpha * sum;
        }
    }
}


void tensor_add_bias_inplace(Tensor* tensor, const Tensor* bias) {
    if (!tensor || !bias) return;

    if (tensor->ndim != 2 || bias->ndim != 2 || bias->shape[0] != 1 || bias->shape[1] != tensor->shape[1]) {
        printf("Error: tensor_add_bias_inplace requires 2D tensor and 1D bias vector\n");
        return;
    }

    int batch_size = tensor->shape[0];
    int feature_size = tensor->shape[1];
    float* tensor_data = tensor->data;
    const float* bias_data = bias->data;

    #pragma omp parallel for schedule(static) if (batch_size > 4)
    for (int b = 0; b < batch_size; ++b) {
        float* row_ptr = &tensor_data[b * feature_size];
        int f = 0;

        for (; f <= feature_size - 8; f += 8) {
            __m256 tensor_vec = _mm256_loadu_ps(&row_ptr[f]);
            __m256 bias_vec = _mm256_loadu_ps(&bias_data[f]);
            tensor_vec = _mm256_add_ps(tensor_vec, bias_vec);
            _mm256_storeu_ps(&row_ptr[f], tensor_vec);
        }

        for (; f < feature_size; ++f) {
            row_ptr[f] += bias_data[f];
        }
    }
}

Tensor* tensor_sum_axis(const Tensor* tensor, int axis) {
    if (!tensor || tensor->ndim != 2 || (axis != 0 && axis != 1)) {
        return NULL;
    }

    int out_shape[2];
    if (axis == 0) {
        out_shape[0] = 1;
        out_shape[1] = tensor->shape[1];
    } else {
        out_shape[0] = tensor->shape[0];
        out_shape[1] = 1;
    }

    Tensor* result = tensor_create_zero(out_shape, 2);
    if (!result) return NULL;

    int batch_size = tensor->shape[0];
    int feature_size = tensor->shape[1];
    const float* tensor_data = tensor->data;
    float* result_data = result->data;

    if (axis == 0) {
        #pragma omp parallel for schedule(static) if (batch_size > 8)
        for (int f = 0; f < feature_size; ++f) {
            float sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                sum += tensor_data[b * feature_size + f];
            }
            result_data[f] = sum;
        }
    } else {
        for (int b = 0; b < batch_size; ++b) {
            float sum = 0.0f;
            int f = 0;
            for (; f <= feature_size - 8; f += 8) {
                __m256 vec = _mm256_loadu_ps(&tensor_data[b * feature_size + f]);
                __m128 hi = _mm256_extractf128_ps(vec, 1);
                __m128 lo = _mm256_castps256_ps128(vec);
                __m128 sum128 = _mm_add_ps(hi, lo);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum += _mm_cvtss_f32(sum128);
            }
            for (; f < feature_size; ++f) {
                sum += tensor_data[b * feature_size + f];
            }
            result_data[b] = sum;
        }
    }

    return result;
}

void tensor_outer_product_accumulate(Tensor* result, const Tensor* a, const Tensor* b) {
    if (!result || !a || !b) return;

    if (result->ndim != 2 || a->ndim != 2 || b->ndim != 2) {
        printf("Error: tensor_outer_product_accumulate requires 2D tensors\n");
        return;
    }

    if (result->shape[0] != a->shape[1] || result->shape[1] != b->shape[1] ||
        a->shape[0] != b->shape[0]) {
        printf("Error: Dimension mismatch in tensor_outer_product_accumulate\n");
        return;
    }

    int batch_size = a->shape[0];
    int output_size = result->shape[0];
    int input_size = result->shape[1];

    tensor_sgemm_simd(output_size, input_size, batch_size,
                     1.0f, a->data, a->shape[1],
                     b->data, b->shape[1],
                     1.0f, result->data, input_size);
}

Tensor* tensor_transpose(const Tensor* tensor) {
    if (!tensor) return NULL;

    if (tensor->ndim != 2) {
        printf("Error: tensor_transpose currently only supports 2D tensors\n");
        return NULL;
    }

    int new_shape[2] = {tensor->shape[1], tensor->shape[0]};
    Tensor* result = tensor_create(new_shape, 2);
    if (!result) return NULL;

    int rows = tensor->shape[0];
    int cols = tensor->shape[1];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result->data[j * rows + i] = tensor->data[i * cols + j];
        }
    }

    return result;
}

void tensor_scale_inplace(Tensor* tensor, float scale) {
    if (!tensor || !tensor->data) return;

    int size = tensor->size;
    float* data = tensor->data;

    int i = 0;
    __m256 scale_vec = _mm256_set1_ps(scale);

    for (; i <= size - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&data[i]);
        vec = _mm256_mul_ps(vec, scale_vec);
        _mm256_storeu_ps(&data[i], vec);
    }

    for (; i < size; i++) {
        data[i] *= scale;
    }
}

void tensor_free(Tensor* tensor) {
    if (tensor) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
    }
}

int tensor_save_binary(const Tensor* tensor, FILE* f) {
    if (!tensor || !f) return 0;

    if (fwrite(&tensor->size, sizeof(int), 1, f) != 1) return 0;
    if (fwrite(&tensor->ndim, sizeof(int), 1, f) != 1) return 0;

    if (fwrite(tensor->shape, sizeof(int), tensor->ndim, f) != (size_t)tensor->ndim) return 0;

    if (fwrite(tensor->data, sizeof(float), tensor->size, f) != (size_t)tensor->size) return 0;

    return 1;
}

int tensor_load_binary(Tensor* tensor, FILE* f) {
    if (!tensor || !f) return 0;

    int size, ndim;
    if (fread(&size, sizeof(int), 1, f) != 1) return 0;
    if (fread(&ndim, sizeof(int), 1, f) != 1) return 0;

    if (tensor->size != size || tensor->ndim != ndim) {
        fprintf(stderr, "Tensor dimensions don't match: expected size=%d ndim=%d, got size=%d ndim=%d\n",
                tensor->size, tensor->ndim, size, ndim);
        return 0;
    }

    int* shape = (int*)malloc(sizeof(int) * ndim);
    if (!shape) return 0;

    if (fread(shape, sizeof(int), ndim, f) != (size_t)ndim) {
        free(shape);
        return 0;
    }

    for (int i = 0; i < ndim; i++) {
        if (tensor->shape[i] != shape[i]) {
            fprintf(stderr, "Tensor shape doesn't match at dimension %d: expected %d, got %d\n",
                    i, tensor->shape[i], shape[i]);
            free(shape);
            return 0;
        }
    }

    free(shape);

    if (fread(tensor->data, sizeof(float), size, f) != (size_t)size) return 0;

    return 1;
}