#include "nn/nn/silu.h"
#include "nn/core/tensor.h"
#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

static inline __m256 exp256_ps(__m256 x) {
    // Temporary stable implementation using scalar expf for correctness
    // TODO: Replace with optimized vectorized exp implementation (e.g., Intel SVML)
    // For some unknown reason, the SIMD part isn't working correctly
    float buffer[8];
    _mm256_storeu_ps(buffer, x);
    for (int i = 0; i < 8; ++i) {
        buffer[i] = expf(buffer[i]);
    }
    return _mm256_loadu_ps(buffer);
}

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
static inline void silu_simd(float* output, const float* input, size_t size) {
    size_t i = 0;
    size_t vector_count = size / 8;
    __m256 one_vec = _mm256_set1_ps(1.0f);

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&input[i]);

        // Compute exp(-x)
        __m256 neg_x_vec = _mm256_sub_ps(_mm256_setzero_ps(), x_vec);
        __m256 exp_neg_x_vec = exp256_ps(neg_x_vec);

        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        __m256 denominator_vec = _mm256_add_ps(one_vec, exp_neg_x_vec);
        __m256 sigmoid_vec = _mm256_div_ps(one_vec, denominator_vec);

        // Compute SiLU(x) = x * sigmoid(x)
        __m256 result_vec = _mm256_mul_ps(x_vec, sigmoid_vec);

        _mm256_storeu_ps(&output[i], result_vec);
    }

    for (; i < size; ++i) {
        float x = input[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid_x;
    }
}

// Gradient of SiLU: d/dx SiLU(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
static inline void silu_grad_simd(float* output, const float* input, const float* grad_output, size_t size) {
    size_t i = 0;
    size_t vector_count = size / 8;
    __m256 one_vec = _mm256_set1_ps(1.0f);

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(&input[i]);
        __m256 grad_vec = _mm256_loadu_ps(&grad_output[i]);

        // Compute exp(-x)
        __m256 neg_x_vec = _mm256_sub_ps(_mm256_setzero_ps(), x_vec);
        __m256 exp_neg_x_vec = exp256_ps(neg_x_vec);

        // Compute sigmoid(x) = 1 / (1 + exp(-x))
        __m256 denominator_vec = _mm256_add_ps(one_vec, exp_neg_x_vec);
        __m256 sigmoid_vec = _mm256_div_ps(one_vec, denominator_vec);

        // Compute 1 - sigmoid(x)
        __m256 one_minus_sigmoid_vec = _mm256_sub_ps(one_vec, sigmoid_vec);

        // Compute sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        __m256 sigmoid_derivative_vec = _mm256_mul_ps(sigmoid_vec, one_minus_sigmoid_vec);

        // Compute x * sigmoid'(x)
        __m256 x_times_sigmoid_derivative_vec = _mm256_mul_ps(x_vec, sigmoid_derivative_vec);

        // Compute gradient = sigmoid(x) + x * sigmoid'(x)
        __m256 gradient_vec = _mm256_add_ps(sigmoid_vec, x_times_sigmoid_derivative_vec);

        __m256 result_vec = _mm256_mul_ps(gradient_vec, grad_vec);

        _mm256_storeu_ps(&output[i], result_vec);
    }

    for (; i < size; ++i) {
        float x = input[i];
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        float sigmoid_derivative = sigmoid_x * (1.0f - sigmoid_x);
        float gradient = sigmoid_x + x * sigmoid_derivative;
        output[i] = gradient * grad_output[i];
    }
}

Tensor* silu(Tensor* input) {
    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    silu_simd(output->data, input->data, input->size);
    return output;
}

Tensor* silu_grad(Tensor* input, Tensor* grad_output) {
    if (!input || !grad_output) return NULL;

    if (input->ndim != grad_output->ndim ||
        memcmp(input->shape, grad_output->shape, input->ndim * sizeof(int)) != 0) {
        printf("Error: Input and gradient dimensions don't match for SiLU gradient\n");
        printf("Input: ndim=%d, shape=[", input->ndim);
        for (int i = 0; i < input->ndim; i++) {
            printf("%d", input->shape[i]);
            if (i < input->ndim - 1) printf(",");
        }
        printf("]\n");
        printf("Grad output: ndim=%d, shape=[", grad_output->ndim);
        for (int i = 0; i < grad_output->ndim; i++) {
            printf("%d", grad_output->shape[i]);
            if (i < grad_output->ndim - 1) printf(",");
        }
        printf("]\n");
        return NULL;
    }

    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    silu_grad_simd(output->data, input->data, grad_output->data, input->size);
    return output;
}
