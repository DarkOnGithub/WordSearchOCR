#define _POSIX_C_SOURCE 199309L
#include "nn/layers/batch_norm2d.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

BatchNorm2D* batch_norm2d_create(int num_features, float momentum, float epsilon) {
    if (num_features <= 0) {
        fprintf(stderr, "Invalid num_features for BatchNorm2D: %d\n", num_features);
        return NULL;
    }

    BatchNorm2D* layer = (BatchNorm2D*)malloc(sizeof(BatchNorm2D));
    if (!layer) {
        fprintf(stderr, "Failed to allocate BatchNorm2D layer\n");
        return NULL;
    }

    layer->num_features = num_features;
    layer->momentum = momentum;
    layer->epsilon = epsilon;
    layer->training = 1;

    int param_shape[1] = {num_features};

    Tensor* gamma = tensor_create_ones(param_shape, 1);
    Tensor* beta = tensor_create_zero(param_shape, 1);
    layer->layer_grad = layer_grad_create(gamma, beta);

    if (!layer->layer_grad) {
        fprintf(stderr, "Failed to create layer gradients for BatchNorm2D\n");
        tensor_free(gamma);
        tensor_free(beta);
        free(layer);
        return NULL;
    }


    layer->running_mean = tensor_create_zero(param_shape, 1);
    layer->running_var = tensor_create_ones(param_shape, 1);

    layer->input_cache = NULL;
    layer->normalized_cache = NULL;
    layer->std_cache = NULL;
    layer->var_cache = NULL;
    layer->mean_cache = NULL;

    return layer;
}

void batch_norm2d_free(BatchNorm2D* layer) {
    if (!layer) return;

    layer_grad_free(layer->layer_grad);
    tensor_free(layer->running_mean);
    tensor_free(layer->running_var);

    if (layer->input_cache) tensor_free(layer->input_cache);
    if (layer->normalized_cache) tensor_free(layer->normalized_cache);
    if (layer->std_cache) tensor_free(layer->std_cache);
    if (layer->var_cache) tensor_free(layer->var_cache);
    if (layer->mean_cache) tensor_free(layer->mean_cache);

    free(layer);
}

void batch_norm2d_output_free(BatchNorm2DOutput* result) {
    if (!result) return;
    tensor_free(result->output);
    free(result);
}

void batch_norm2d_backward_output_free(BatchNorm2DBackwardOutput* result) {
    if (!result) return;
    tensor_free(result->input_grad);
    free(result);
}

void batch_norm2d_set_training(BatchNorm2D* layer, int training) {
    layer->training = training;
}

static inline void batch_norm2d_compute_mean_var(const float* input, int batch_size,
                                               int num_features, int height, int width,
                                               float* mean, float* var) {

    for (int c = 0; c < num_features; ++c) {
        double sum = 0.0;
        double sum_sq = 0.0;
        int count = 0;

        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int idx = ((b * num_features + c) * height + h) * width + w;
                    float val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
        }

        mean[c] = sum / count;
        var[c] = (sum_sq / count) - (mean[c] * mean[c]);
    }
}

static inline void batch_norm2d_normalize_simd(float* output, const float* input,
                                             const float* mean, const float* std,
                                             int batch_size, int num_features,
                                             int height, int width) {
    int spatial_size = height * width;

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            float mean_val = mean[c];
            float std_val = std[c];
            int base_idx = (b * num_features + c) * spatial_size;

            int hw = 0;
            for (; hw <= spatial_size - 8; hw += 8) {
                __m256 input_vec = _mm256_loadu_ps(&input[base_idx + hw]);
                __m256 mean_vec = _mm256_set1_ps(mean_val);
                __m256 std_vec = _mm256_set1_ps(std_val);

                __m256 normalized = _mm256_div_ps(_mm256_sub_ps(input_vec, mean_vec), std_vec);
                _mm256_storeu_ps(&output[base_idx + hw], normalized);
            }

            for (; hw < spatial_size; ++hw) {
                output[base_idx + hw] = (input[base_idx + hw] - mean_val) / std_val;
            }
        }
    }
}

BatchNorm2DOutput* batch_norm2d_forward(BatchNorm2D* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    if (input->ndim != 4 || input->shape[1] != layer->num_features) {
        printf("Error: Input tensor must be 4D with num_features %d\n", layer->num_features);
        return NULL;
    }

    int batch_size = input->shape[0];
    int num_features = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    int output_shape[4] = {batch_size, num_features, height, width};
    Tensor* output = tensor_create(output_shape, 4);
    if (!output) return NULL;

    float* batch_mean = (float*)malloc(num_features * sizeof(float));
    float* batch_var = (float*)malloc(num_features * sizeof(float));
    float* batch_std = (float*)malloc(num_features * sizeof(float));
    if (!batch_mean || !batch_var || !batch_std) {
        fprintf(stderr, "Failed to allocate batch statistics buffers\n");
        tensor_free(output);
        free(batch_mean);
        free(batch_var);
        free(batch_std);
        return NULL;
    }

    if (layer->training) {
        batch_norm2d_compute_mean_var(input->data, batch_size, num_features, height, width,
                                    batch_mean, batch_var);

        #pragma omp parallel for schedule(static) if (num_features > 4)
        for (int c = 0; c < num_features; ++c) {
            layer->running_mean->data[c] = (1.0f - layer->momentum) * layer->running_mean->data[c] +
                                          layer->momentum * batch_mean[c];
            layer->running_var->data[c] = (1.0f - layer->momentum) * layer->running_var->data[c] +
                                         layer->momentum * batch_var[c];
        }

        if (layer->input_cache) tensor_free(layer->input_cache);
        layer->input_cache = tensor_create(input->shape, input->ndim);
        if (layer->input_cache) {
            memcpy(layer->input_cache->data, input->data, input->size * sizeof(float));
        }

    } else {
        memcpy(batch_mean, layer->running_mean->data, num_features * sizeof(float));
        memcpy(batch_var, layer->running_var->data, num_features * sizeof(float));
    }

    #pragma omp parallel for schedule(static) if (num_features > 4)
    for (int c = 0; c < num_features; ++c) {
        batch_std[c] = sqrtf(batch_var[c] + layer->epsilon);
    }

    if (layer->training) {
        if (layer->mean_cache) tensor_free(layer->mean_cache);
        layer->mean_cache = tensor_create(&num_features, 1);
        if (layer->mean_cache) {
            memcpy(layer->mean_cache->data, batch_mean, num_features * sizeof(float));
        }

        if (layer->var_cache) tensor_free(layer->var_cache);
        layer->var_cache = tensor_create(&num_features, 1);
        if (layer->var_cache) {
            memcpy(layer->var_cache->data, batch_var, num_features * sizeof(float));
        }

        if (layer->std_cache) tensor_free(layer->std_cache);
        layer->std_cache = tensor_create(&num_features, 1);
        if (layer->std_cache) {
            memcpy(layer->std_cache->data, batch_std, num_features * sizeof(float));
        }
    }

    batch_norm2d_normalize_simd(output->data, input->data, batch_mean, batch_std,
                              batch_size, num_features, height, width);

    if (layer->training) {
        if (layer->normalized_cache) tensor_free(layer->normalized_cache);
        layer->normalized_cache = tensor_create(output->shape, output->ndim);
        if (layer->normalized_cache) {
            memcpy(layer->normalized_cache->data, output->data, output->size * sizeof(float));
        }
    }

    #pragma omp parallel for collapse(2) schedule(static) if (batch_size * num_features > 4)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            float gamma = layer->layer_grad->weights->data[c];
            float beta = layer->layer_grad->biases->data[c];
            int spatial_size = height * width;
            int base_idx = (b * num_features + c) * spatial_size;

            for (int hw = 0; hw < spatial_size; ++hw) {
                output->data[base_idx + hw] = gamma * output->data[base_idx + hw] + beta;
            }
        }
    }

    free(batch_mean);
    free(batch_var);
    free(batch_std);

    BatchNorm2DOutput* result = (BatchNorm2DOutput*)malloc(sizeof(BatchNorm2DOutput));
    if (!result) {
        tensor_free(output);
        return NULL;
    }
    result->output = output;
    result->layer = layer;

    return result;
}

BatchNorm2DBackwardOutput* batch_norm2d_backward(BatchNorm2D* layer, BatchNorm2DOutput* forward_result, Tensor* output_grad) {
    if (!layer || !forward_result || !output_grad || !layer->training) return NULL;

    if (output_grad->ndim != 4 || output_grad->shape[1] != layer->num_features) {
        fprintf(stderr, "BatchNorm2D backward: output_grad must be 4D with correct num_features\n");
        return NULL;
    }

    int batch_size = output_grad->shape[0];
    int num_features = output_grad->shape[1];
    int height = output_grad->shape[2];
    int width = output_grad->shape[3];
    int spatial_size = height * width;

    int input_grad_shape[4] = {batch_size, num_features, height, width};
    Tensor* input_grad = tensor_create(input_grad_shape, 4);
    if (!input_grad) return NULL;

    const float* gamma = layer->layer_grad->weights->data;

    float* gamma_grad = (float*)calloc(num_features, sizeof(float));
    float* beta_grad = (float*)calloc(num_features, sizeof(float));
    if (!gamma_grad || !beta_grad) {
        tensor_free(input_grad);
        free(gamma_grad);
        free(beta_grad);
        return NULL;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            int base_idx = (b * num_features + c) * spatial_size;
            for (int hw = 0; hw < spatial_size; ++hw) {
                float grad = output_grad->data[base_idx + hw];
                float norm = layer->normalized_cache->data[base_idx + hw];
                gamma_grad[c] += grad * norm;
                beta_grad[c] += grad;
            }
        }
    }

    memcpy(layer->layer_grad->weight_grad->data, gamma_grad, num_features * sizeof(float));
    memcpy(layer->layer_grad->bias_grad->data, beta_grad, num_features * sizeof(float));

    const float* std_data = layer->std_cache->data;
    const float* var_data = layer->var_cache->data;
    const float* mean_data = layer->mean_cache->data;
    const float* input_data = layer->input_cache->data;

    int m = batch_size * spatial_size;
    float inv_m = 1.0f / m;

    float* dL_dvar = (float*)malloc(num_features * sizeof(float));
    float* dL_dmu = (float*)malloc(num_features * sizeof(float));
    if (!dL_dvar || !dL_dmu) {
        free(dL_dvar);
        free(dL_dmu);
        tensor_free(input_grad);
        free(gamma_grad);
        free(beta_grad);
        return NULL;
    }

    for (int c = 0; c < num_features; ++c) {
        float gamma_val = gamma[c];
        float var_val = var_data[c];
        float var_plus_eps = var_val + layer->epsilon;
        float std_val = std_data[c];
        float inv_std = 1.0f / std_val;

        double sum_dx_hat = 0.0;
        double sum_dx_hat_x_minus_mu = 0.0;
        double sum_x_minus_mu = 0.0;

        for (int b = 0; b < batch_size; ++b) {
            int base_idx = (b * num_features + c) * spatial_size;
            float mu_val = mean_data[c];

            for (int hw = 0; hw < spatial_size; ++hw) {
                float grad = output_grad->data[base_idx + hw];
                float dx_hat = gamma_val * grad;
                float x_val = input_data[base_idx + hw];
                float x_minus_mu = x_val - mu_val;

                sum_dx_hat += dx_hat;
                sum_dx_hat_x_minus_mu += dx_hat * x_minus_mu;
                sum_x_minus_mu += x_minus_mu;
            }
        }

        dL_dvar[c] = sum_dx_hat_x_minus_mu * (-0.5f) / (var_plus_eps * sqrtf(var_plus_eps));
        dL_dmu[c] = -sum_dx_hat * inv_std + dL_dvar[c] * (-2.0f * sum_x_minus_mu) * inv_m;
    }

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < num_features; ++c) {
            float gamma_val = gamma[c];
            float std_val = std_data[c];
            float inv_std = 1.0f / std_val;
            float dvar_term = 2.0f * inv_m * dL_dvar[c];
            float dmu_term = inv_m * dL_dmu[c];

            int base_idx = (b * num_features + c) * spatial_size;
            float mu_val = mean_data[c];

            int hw = 0;
            for (; hw <= spatial_size - 8; hw += 8) {
                __m256 grad_vec = _mm256_loadu_ps(&output_grad->data[base_idx + hw]);
                __m256 x_vec = _mm256_loadu_ps(&input_data[base_idx + hw]);

                __m256 gamma_vec = _mm256_set1_ps(gamma_val);
                __m256 dx_hat_vec = _mm256_mul_ps(gamma_vec, grad_vec);

                __m256 mu_vec = _mm256_set1_ps(mu_val);
                __m256 x_minus_mu_vec = _mm256_sub_ps(x_vec, mu_vec);

                __m256 grad_term1_vec = _mm256_mul_ps(dx_hat_vec, _mm256_set1_ps(inv_std));
                __m256 grad_term2_vec = _mm256_mul_ps(_mm256_set1_ps(dvar_term), x_minus_mu_vec);
                __m256 grad_term3_vec = _mm256_set1_ps(dmu_term);

                __m256 result_vec = _mm256_add_ps(_mm256_add_ps(grad_term1_vec, grad_term2_vec), grad_term3_vec);

                _mm256_storeu_ps(&input_grad->data[base_idx + hw], result_vec);
            }

            for (; hw < spatial_size; ++hw) {
                float grad = output_grad->data[base_idx + hw];
                float dx_hat = gamma_val * grad;
                float x_val = input_data[base_idx + hw];
                float x_minus_mu = x_val - mu_val;

                float grad_term1 = dx_hat * inv_std;
                float grad_term2 = dvar_term * x_minus_mu;
                float grad_term3 = dmu_term;

                input_grad->data[base_idx + hw] = grad_term1 + grad_term2 + grad_term3;
            }
        }
    }

    free(dL_dvar);
    free(dL_dmu);

    free(gamma_grad);
    free(beta_grad);

    BatchNorm2DBackwardOutput* result = (BatchNorm2DBackwardOutput*)malloc(sizeof(BatchNorm2DBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }
    result->input_grad = input_grad;

    return result;
}
