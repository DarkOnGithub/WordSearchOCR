#include "nn/nn/relu.h"
#include "nn/core/tensor.h"
#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>

static inline void relu_simd(float* output, const float* input, size_t size) {
    size_t i = 0;
    size_t vector_count = size / 8;
    __m256 zero_vec = _mm256_setzero_ps();

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 input_vec = _mm256_loadu_ps(&input[i]);
        __m256 result_vec = _mm256_max_ps(input_vec, zero_vec);
        _mm256_storeu_ps(&output[i], result_vec);
    }

    for (; i < size; ++i) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

static inline void relu_grad_simd(float* output, const float* input, const float* grad_output, size_t size) {
    size_t i = 0;
    size_t vector_count = size / 8;
    __m256 zero_vec = _mm256_setzero_ps();

    for (i = 0; i < vector_count * 8; i += 8) {
        __m256 input_vec = _mm256_loadu_ps(&input[i]);
        __m256 grad_vec = _mm256_loadu_ps(&grad_output[i]);

        __m256 mask = _mm256_cmp_ps(input_vec, zero_vec, _CMP_GT_OQ);
        __m256 result_vec = _mm256_and_ps(grad_vec, mask);

        _mm256_storeu_ps(&output[i], result_vec);
    }

    for (; i < size; ++i) {
        output[i] = (input[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

Tensor* relu(Tensor* input) {
    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    relu_simd(output->data, input->data, input->size);
    return output;
}

Tensor* relu_grad(Tensor* input, Tensor* grad_output) {
    if (!input || !grad_output) return NULL;

    if (input->ndim != grad_output->ndim ||
        memcmp(input->shape, grad_output->shape, input->ndim * sizeof(int)) != 0) {
        printf("Error: Input and gradient dimensions don't match for ReLU gradient\n");
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

    relu_grad_simd(output->data, input->data, grad_output->data, input->size);
    return output;
}