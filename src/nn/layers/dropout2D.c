#include "nn/layers/dropout2D.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include "nn/core/utils.h"
#include <string.h>

static inline void broadcast_mask_to_tensor(float* __restrict output,
                                                 const float* __restrict input,
                                                 const float* __restrict mask,
                                                 int batch_size, int channels,
                                                 int height, int width) {
    // Broadcast mask from (batch_size, channels, 1, 1) to (batch_size, channels, height, width)
    const int spatial_size = height * width;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            const int mask_idx = b * channels + c;
            const __m256 mask_vec = _mm256_set1_ps(mask[mask_idx]);

            const int input_base = (b * channels + c) * spatial_size;
            const int output_base = (b * channels + c) * spatial_size;

            int w = 0;
            for (; w <= spatial_size - 8; w += 8) {
                __m256 input_vec = _mm256_loadu_ps(&input[input_base + w]);
                __m256 result_vec = _mm256_mul_ps(input_vec, mask_vec);
                _mm256_storeu_ps(&output[output_base + w], result_vec);
            }

            for (; w < spatial_size; ++w) {
                output[output_base + w] = input[input_base + w] * mask[mask_idx];
            }
        }
    }
}

static inline void broadcast_mask_to_grad(float* __restrict input_grad,
                                               const float* __restrict output_grad,
                                               const float* __restrict mask,
                                               int batch_size, int channels,
                                               int height, int width) {
    // Broadcast mask from (batch_size, channels, 1, 1) to (batch_size, channels, height, width)
    const int spatial_size = height * width;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            const int mask_idx = b * channels + c;
            const __m256 mask_vec = _mm256_set1_ps(mask[mask_idx]);

            const int grad_base = (b * channels + c) * spatial_size;

            int w = 0;
            for (; w <= spatial_size - 8; w += 8) {
                __m256 grad_vec = _mm256_loadu_ps(&output_grad[grad_base + w]);
                __m256 result_vec = _mm256_mul_ps(grad_vec, mask_vec);
                _mm256_storeu_ps(&input_grad[grad_base + w], result_vec);
            }

            for (; w < spatial_size; ++w) {
                input_grad[grad_base + w] = output_grad[grad_base + w] * mask[mask_idx];
            }
        }
    }
}

Dropout2D* dropout2d_create(float dropout_rate) {
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        printf("Error: Dropout2D rate must be in range [0.0, 1.0)\n");
        return NULL;
    }

    Dropout2D* layer = (Dropout2D*)malloc(sizeof(Dropout2D));
    if (!layer) return NULL;
    layer->dropout_rate = dropout_rate;
    layer->training = true;
    return layer;
}

void dropout2d_free(Dropout2D* layer) {
    if (layer) {
        free(layer);
    }
}


Dropout2DOutput* dropout2d_forward(Dropout2D* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    Tensor* mask = NULL;

    if (layer->training && layer->dropout_rate > 0.0f) {
        // Create mask with shape (batch_size, channels, 1, 1)
        int mask_shape[4] = {input->shape[0], input->shape[1], 1, 1};
        mask = tensor_create(mask_shape, 4);
        if (!mask) {
            tensor_free(output);
            return NULL;
        }

        // Generate dropout mask for each channel in each batch
        for (int b = 0; b < input->shape[0]; ++b) {
            for (int c = 0; c < input->shape[1]; ++c) {
                float rand_val = random_float();
                int mask_idx = b * input->shape[1] + c;

                if (rand_val < layer->dropout_rate) {
                    // Drop this entire channel
                    mask->data[mask_idx] = 0.0f;
                } else {
                    // Keep this channel, scale by (1/(1-dropout_rate)) for variance preservation
                    mask->data[mask_idx] = 1.0f / (1.0f - layer->dropout_rate);
                }
            }
        }

        broadcast_mask_to_tensor(output->data, input->data, mask->data,
                                     input->shape[0], input->shape[1],
                                     input->shape[2], input->shape[3]);
    } else {
        // Inference mode or dropout_rate = 0: scale by (1-dropout_rate)
        memcpy(output->data, input->data, input->size * sizeof(float));
        tensor_scale_inplace(output, 1.0f - layer->dropout_rate);
    }

    Dropout2DOutput* result = (Dropout2DOutput*)malloc(sizeof(Dropout2DOutput));
    if (!result) {
        tensor_free(output);
        if (mask) tensor_free(mask);
        return NULL;
    }

    result->output = output;
    result->mask = mask;

    return result;
}

void dropout2d_output_free(Dropout2DOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        if (result->mask) tensor_free(result->mask);
        free(result);
    }
}

Dropout2DBackwardOutput* dropout2d_backward(Dropout2D* layer, Dropout2DOutput* forward_result,
                                           Tensor* output_grad) {
    if (!layer || !forward_result || !forward_result->output || !output_grad) return NULL;

    if (output_grad->ndim != forward_result->output->ndim ||
        output_grad->shape[0] != forward_result->output->shape[0] ||
        output_grad->shape[1] != forward_result->output->shape[1] ||
        output_grad->shape[2] != forward_result->output->shape[2] ||
        output_grad->shape[3] != forward_result->output->shape[3]) {
        printf("Error: Output gradient dimensions don't match forward output dimensions got %d x %d x %d x %d expected %d x %d x %d x %d\n",
               output_grad->shape[0], output_grad->shape[1], output_grad->shape[2], output_grad->shape[3],
               forward_result->output->shape[0], forward_result->output->shape[1], forward_result->output->shape[2], forward_result->output->shape[3]);
        return NULL;
    }

    Tensor* input_grad = tensor_create(output_grad->shape, output_grad->ndim);
    if (!input_grad) return NULL;

    if (layer->training && forward_result->mask) {
        // Training mode: use the dropout mask to route gradients
        // Broadcast mask from (batch_size, channels, 1, 1) to full tensor shape
        broadcast_mask_to_grad(input_grad->data, output_grad->data, forward_result->mask->data,
                                   output_grad->shape[0], output_grad->shape[1],
                                   output_grad->shape[2], output_grad->shape[3]);
    } else {
        // Inference mode or no mask: scale gradients by (1-dropout_rate)
        memcpy(input_grad->data, output_grad->data, output_grad->size * sizeof(float));
        tensor_scale_inplace(input_grad, 1.0f - layer->dropout_rate);
    }

    Dropout2DBackwardOutput* result = (Dropout2DBackwardOutput*)malloc(sizeof(Dropout2DBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void dropout2d_backward_output_free(Dropout2DBackwardOutput* result) {
    if (result) {
        if (result->input_grad) tensor_free(result->input_grad);
        free(result);
    }
}

