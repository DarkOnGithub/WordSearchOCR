#include "nn/layers/cross_entropy_loss.h"
#include "nn/core/tensor.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <string.h>

static inline float horizontal_max_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 max1 = _mm_max_ps(lo, hi);
    __m128 max2 = _mm_movehl_ps(max1, max1);
    __m128 max3 = _mm_max_ps(max1, max2);
    __m128 max4 = _mm_shuffle_ps(max3, max3, 0x55);
    __m128 max_final = _mm_max_ps(max3, max4);
    return _mm_cvtss_f32(max_final);
}

static inline float horizontal_sum_avx(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    __m128 sum2 = _mm_movehl_ps(sum, sum);
    __m128 sum3 = _mm_add_ps(sum, sum2);
    __m128 sum4 = _mm_shuffle_ps(sum3, sum3, 0x55);
    __m128 sum_final = _mm_add_ps(sum3, sum4);
    return _mm_cvtss_f32(sum_final);
}

CrossEntropyLoss* cross_entropy_loss_create() {
    CrossEntropyLoss* loss = (CrossEntropyLoss*)malloc(sizeof(CrossEntropyLoss));
    if (!loss) {
        fprintf(stderr, "Error: Failed to allocate memory for CrossEntropyLoss\n");
        return NULL;
    }
    return loss;
}

void cross_entropy_loss_free(CrossEntropyLoss* loss) {
    if (loss) {
        free(loss);
    }
}

// Compute softmax along the last dimension
// Handles both 2D (batch_size, num_classes) and 4D inputs
Tensor* softmax(Tensor* input) {
    if (!input) return NULL;

    int batch_size = input->shape[0];
    int num_classes;

    if (input->ndim == 2) {
        num_classes = input->shape[1];
    } else if (input->ndim == 4) {
        num_classes = input->shape[1] * input->shape[2] * input->shape[3];
    } else {
        fprintf(stderr, "Error: softmax expects 2D or 4D input, got %dD\n", input->ndim);
        return NULL;
    }

    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    for (int b = 0; b < batch_size; b++) {
        // Find max value for numerical stability
        __m256 max_vec = _mm256_set1_ps(-INFINITY);
        int i = 0;
        int idx = b * num_classes;

        for (; i <= num_classes - 8; i += 8) {
            __m256 data_vec = _mm256_loadu_ps(&input->data[idx + i]);
            max_vec = _mm256_max_ps(max_vec, data_vec);
        }

        float max_val = horizontal_max_avx(max_vec);

        for (; i < num_classes; i++) {
            if (input->data[idx + i] > max_val) {
                max_val = input->data[idx + i];
            }
        }

        // Compute exp(x - max) and sum with numerical stability
        __m256 sum_vec = _mm256_setzero_ps();
        int idx_base = b * num_classes;

        int j = 0;
        for (; j <= num_classes - 8; j += 8) {
            __m256 input_vec = _mm256_loadu_ps(&input->data[idx_base + j]);
            __m256 max_vec = _mm256_set1_ps(max_val);
            __m256 shifted_vec = _mm256_sub_ps(input_vec, max_vec);

            float exp_vals[8];
            _mm256_storeu_ps(exp_vals, shifted_vec);
            for (int k = 0; k < 8; k++) {
                // Clamp to prevent overflow/underflow
                if (exp_vals[k] > 80.0f) exp_vals[k] = 80.0f;
                if (exp_vals[k] < -80.0f) exp_vals[k] = -80.0f;
                exp_vals[k] = expf(exp_vals[k]);
            }

            __m256 exp_vec = _mm256_loadu_ps(exp_vals);
            _mm256_storeu_ps(&output->data[idx_base + j], exp_vec);
            sum_vec = _mm256_add_ps(sum_vec, exp_vec);
        }

        float sum_exp = horizontal_sum_avx(sum_vec);

        for (; j < num_classes; j++) {
            float shifted_val = input->data[idx_base + j] - max_val;
            // Clamp to prevent overflow/underflow
            if (shifted_val > 80.0f) shifted_val = 80.0f;
            if (shifted_val < -80.0f) shifted_val = -80.0f;
            float exp_val = expf(shifted_val);
            output->data[idx_base + j] = exp_val;
            sum_exp += exp_val;
        }

        // Ensure sum_exp is not zero (shouldn't happen with clamping, but safety check)
        if (sum_exp < 1e-20f) {
            sum_exp = 1e-20f;
        }

        // Normalize by sum
        __m256 sum_inv_vec = _mm256_set1_ps(1.0f / sum_exp);
        int idx_norm = b * num_classes;
        int m = 0;

        for (; m <= num_classes - 8; m += 8) {
            __m256 exp_vec = _mm256_loadu_ps(&output->data[idx_norm + m]);
            __m256 normalized_vec = _mm256_mul_ps(exp_vec, sum_inv_vec);
            _mm256_storeu_ps(&output->data[idx_norm + m], normalized_vec);
        }

        for (; m < num_classes; m++) {
            output->data[idx_norm + m] /= sum_exp;
        }
    }

    return output;
}

CrossEntropyOutput* cross_entropy_loss_forward(CrossEntropyLoss* loss, Tensor* input, Tensor* targets) {
    if (!loss || !input || !targets) {
        fprintf(stderr, "Error: NULL input to cross_entropy_loss_forward\n");
        return NULL;
    }

    int batch_size = input->shape[0];
    int num_classes;

    if (input->ndim == 2) {
        num_classes = input->shape[1];
    } else if (input->ndim == 4) {
        num_classes = input->shape[1] * input->shape[2] * input->shape[3];
    } else {
        fprintf(stderr, "Error: cross_entropy_loss_forward expects 2D or 4D input, got %dD\n", input->ndim);
        return NULL;
    }

    if (targets->shape[0] != batch_size || targets->size != batch_size) {
        fprintf(stderr, "Error: Target tensor must have batch_size elements\n");
        return NULL;
    }

    Tensor* softmax_output = softmax(input);
    if (!softmax_output) {
        return NULL;
    }

    // Compute negative log likelihood
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        int target_class = (int)targets->data[b];
        if (target_class < 0 || target_class >= num_classes) {
            fprintf(stderr, "Error: Target class %d out of range [0, %d]\n", target_class, num_classes - 1);
            tensor_free(softmax_output);
            return NULL;
        }

        int idx = b * num_classes + target_class;
        float prob = softmax_output->data[idx];

        // Add small epsilon for numerical stability
        if (prob < 1e-7f) {
            prob = 1e-7f;
        }

        total_loss += -logf(prob);
    }

    float avg_loss = total_loss / batch_size;

    CrossEntropyOutput* result = (CrossEntropyOutput*)malloc(sizeof(CrossEntropyOutput));
    if (!result) {
        fprintf(stderr, "Error: Failed to allocate memory for CrossEntropyOutput\n");
        tensor_free(softmax_output);
        return NULL;
    }

    result->loss = avg_loss;
    result->softmax_output = softmax_output;
    result->targets = tensor_create(targets->shape, targets->ndim);
    if (!result->targets) {
        fprintf(stderr, "Error: Failed to allocate memory for targets copy\n");
        tensor_free(softmax_output);
        free(result);
        return NULL;
    }
    memcpy(result->targets->data, targets->data, targets->size * sizeof(float));
    return result;
}

void cross_entropy_result_free(CrossEntropyOutput* result) {
    if (result) {
        if (result->softmax_output) {
            tensor_free(result->softmax_output);
        }
        if (result->targets) {
            tensor_free(result->targets);
        }
        free(result);
    }
}

CrossEntropyBackwardOutput* cross_entropy_loss_backward(CrossEntropyLoss* loss,
                                                        CrossEntropyOutput* forward_result,
                                                        Tensor* output_grad) {
    if (!loss || !forward_result || !forward_result->softmax_output || !forward_result->targets) {
        fprintf(stderr, "Error: NULL input to cross_entropy_loss_backward\n");
        return NULL;
    }

    int batch_size = forward_result->softmax_output->shape[0];
    int num_classes;

    if (forward_result->softmax_output->ndim == 2) {
        num_classes = forward_result->softmax_output->shape[1];
    } else if (forward_result->softmax_output->ndim == 4) {
        num_classes = forward_result->softmax_output->shape[1] *
                     forward_result->softmax_output->shape[2] *
                     forward_result->softmax_output->shape[3];
    } else {
        fprintf(stderr, "Error: cross_entropy_loss_backward expects 2D or 4D input, got %dD\n",
                forward_result->softmax_output->ndim);
        return NULL;
    }

    Tensor* input_grad = tensor_create(forward_result->softmax_output->shape,
                                      forward_result->softmax_output->ndim);
    if (!input_grad) {
        return NULL;
    }

    // Initialize gradient to softmax probabilities
    int i = 0;
    for (; i <= input_grad->size - 8; i += 8) {
        __m256 softmax_vec = _mm256_loadu_ps(&forward_result->softmax_output->data[i]);
        _mm256_storeu_ps(&input_grad->data[i], softmax_vec);
    }

    for (; i < input_grad->size; i++) {
        input_grad->data[i] = forward_result->softmax_output->data[i];
    }

    // Subtract one-hot targets: grad = softmax - one_hot(targets)
    for (int b = 0; b < batch_size; b++) {
        int target_class = (int)forward_result->targets->data[b];
        int idx = b * num_classes + target_class;
        input_grad->data[idx] -= 1.0f;
    }

    // Scale by output gradient
    // Note: loss is already averaged over batch in forward pass,
    // so gradients here are for the averaged loss
    float grad_scale = 1.0f;
    if (output_grad && output_grad->size > 0) {
        grad_scale = output_grad->data[0];
    }

    tensor_scale_inplace(input_grad, grad_scale);

    CrossEntropyBackwardOutput* result = (CrossEntropyBackwardOutput*)malloc(sizeof(CrossEntropyBackwardOutput));
    if (!result) {
        fprintf(stderr, "Error: Failed to allocate memory for CrossEntropyBackwardOutput\n");
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void cross_entropy_backward_result_free(CrossEntropyBackwardOutput* result) {
    if (result) {
        if (result->input_grad) {
            tensor_free(result->input_grad);
        }
        free(result);
    }
}


