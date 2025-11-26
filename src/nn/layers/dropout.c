#include "nn/layers/dropout.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include "nn/core/utils.h"

static inline void dropout_apply_grad_mask(float* __restrict input_grad,
                                                const float* __restrict output_grad,
                                                const float* __restrict mask,
                                                size_t size) {
    size_t i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 grad_vec = _mm256_loadu_ps(&output_grad[i]);
        __m256 mask_vec = _mm256_loadu_ps(&mask[i]);
        __m256 result_vec = _mm256_mul_ps(grad_vec, mask_vec);
        _mm256_storeu_ps(&input_grad[i], result_vec);
    }

    for (; i < size; ++i) {
        input_grad[i] = output_grad[i] * mask[i];
    }
}

Dropout* dropout_create(float dropout_rate) {
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        printf("Error: Dropout rate must be in range [0.0, 1.0)\n");
        return NULL;
    }

    Dropout* layer = (Dropout*)malloc(sizeof(Dropout));
    if (!layer) return NULL;

    layer->dropout_rate = dropout_rate;
    layer->training = true;

    srand((unsigned int)time(NULL));

    return layer;
}

void dropout_free(Dropout* layer) {
    if (layer) {
        free(layer);
    }
}


DropoutOutput* dropout_forward(Dropout* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    Tensor* mask = NULL;

    if (layer->training && layer->dropout_rate > 0.0f) {
        mask = tensor_create(input->shape, input->ndim);
        if (!mask) {
            tensor_free(output);
            return NULL;
        }

        for (int i = 0; i < input->size; ++i) {
            float rand_val = random_float();
            if (rand_val < layer->dropout_rate) {
                mask->data[i] = 0.0f;
                output->data[i] = 0.0f;
            } else {
                mask->data[i] = 1.0f / (1.0f - layer->dropout_rate);
                output->data[i] = input->data[i] * mask->data[i];
            }
        }
    } else {
        memcpy(output->data, input->data, input->size * sizeof(float));
        tensor_scale_inplace(output, 1.0f - layer->dropout_rate);
    }

    DropoutOutput* result = (DropoutOutput*)malloc(sizeof(DropoutOutput));
    if (!result) {
        tensor_free(output);
        if (mask) tensor_free(mask);
        return NULL;
    }

    result->output = output;
    result->mask = mask;

    return result;
}

void dropout_output_free(DropoutOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        if (result->mask) tensor_free(result->mask);
        free(result);
    }
}

DropoutBackwardOutput* dropout_backward(Dropout* layer, DropoutOutput* forward_result,
                                       Tensor* output_grad) {
    if (!layer || !forward_result || !forward_result->output || !output_grad) return NULL;

    if (output_grad->ndim != forward_result->output->ndim ||
        memcmp(output_grad->shape, forward_result->output->shape,
               output_grad->ndim * sizeof(int)) != 0) {
        printf("Error: Output gradient dimensions don't match forward output dimensions\n");
        return NULL;
    }

    Tensor* input_grad = tensor_create(output_grad->shape, output_grad->ndim);
    if (!input_grad) return NULL;

    if (layer->training && forward_result->mask) {
        dropout_apply_grad_mask(input_grad->data, output_grad->data,
                                    forward_result->mask->data, output_grad->size);
    } else {
        memcpy(input_grad->data, output_grad->data, output_grad->size * sizeof(float));
        tensor_scale_inplace(input_grad, 1.0f - layer->dropout_rate);
    }

    DropoutBackwardOutput* result = (DropoutBackwardOutput*)malloc(sizeof(DropoutBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void dropout_backward_output_free(DropoutBackwardOutput* result) {
    if (result) {
        if (result->input_grad) tensor_free(result->input_grad);
        free(result);
    }
}


