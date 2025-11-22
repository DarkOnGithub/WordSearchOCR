#include "nn/layers/linear.h"
#include "nn/core/layer_grad.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include "nn/core/init.h"



Linear* linear_create(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) {
        printf("Error: Input and output sizes must be positive\n");
        return NULL;
    }

    Linear* layer = (Linear*)malloc(sizeof(Linear));
    if (!layer) return NULL;

    layer->input_size = input_size;
    layer->output_size = output_size;

    int weight_shape[2] = {input_size, output_size};
    int bias_shape[1] = {output_size};
    Tensor *weights = tensor_create(weight_shape, 2);
    Tensor *biases = tensor_create(bias_shape, 1);
    layer->layer_grad = layer_grad_create(weights, biases);

    if (!layer->layer_grad) {
        tensor_free(weights);
        tensor_free(biases);
        free(layer);
        return NULL;
    }

    init_xavier_uniform(layer->layer_grad->weights);
    memset(layer->layer_grad->biases->data, 0, layer->layer_grad->biases->size * sizeof(float));

    layer->input_cache = NULL;

    return layer;
}

void linear_free(Linear* layer) {
    if (layer) {
        if (layer->layer_grad->weights) tensor_free(layer->layer_grad->weights);
        if (layer->layer_grad->biases) tensor_free(layer->layer_grad->biases);
        if (layer->layer_grad->weight_grad) tensor_free(layer->layer_grad->weight_grad);
        if (layer->layer_grad->bias_grad) tensor_free(layer->layer_grad->bias_grad);
        if (layer->input_cache) tensor_free(layer->input_cache);
        free(layer);
    }
}

LinearOutput* linear_forward(Linear* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    int batch_size;
    int flattened_input_size;

    if (input->ndim == 2) {
        batch_size = input->shape[0];
        flattened_input_size = input->shape[1];
    } else if (input->ndim == 4) {
        batch_size = input->shape[0];
        flattened_input_size = input->shape[1] * input->shape[2] * input->shape[3];
    } else {
        printf("Error: Input must be 2D or 4D tensor\n");
        return NULL;
    }

    if (flattened_input_size != layer->input_size) {
        printf("Error: Input size %d doesn't match layer input size %d\n",
               flattened_input_size, layer->input_size);
        return NULL;
    }

    int output_shape[2] = {batch_size, layer->output_size};
    Tensor* output = tensor_create(output_shape, 2);
    if (!output) return NULL;

    if (layer->input_cache) tensor_free(layer->input_cache);
    int input_cache_shape[2] = {batch_size, flattened_input_size};
    layer->input_cache = tensor_create(input_cache_shape, 2);
    if (!layer->input_cache) {
        tensor_free(output);
        return NULL;
    }

    memcpy(layer->input_cache->data, input->data, input->size * sizeof(float));

    const int M = batch_size;
    const int K = layer->input_size;
    const int N = layer->output_size;

    memset(output->data, 0, output->size * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const float* input_row = &layer->input_cache->data[m * K];
        float* output_row = &output->data[m * N];

        for (int k = 0; k < K; ++k) {
            const float input_val = input_row[k];
            const float* weight_row = &layer->layer_grad->weights->data[k * N];

            __m256 input_broadcast = _mm256_set1_ps(input_val);

            int n = 0;
            for (; n <= N - 8; n += 8) {
                __m256 output_vec = _mm256_loadu_ps(&output_row[n]);
                __m256 weight_vec = _mm256_loadu_ps(&weight_row[n]);
                output_vec = _mm256_fmadd_ps(input_broadcast, weight_vec, output_vec);
                _mm256_storeu_ps(&output_row[n], output_vec);
            }

            for (; n < N; ++n) {
                output_row[n] += input_val * weight_row[n];
            }
        }
    }


    #pragma omp parallel for schedule(static) if (batch_size > 4)
    for (int b = 0; b < batch_size; ++b) {
        float* output_row = &output->data[b * layer->output_size];
        int o = 0;
        for (; o <= layer->output_size - 8; o += 8) {
            __m256 output_vec = _mm256_loadu_ps(&output_row[o]);
            __m256 bias_vec = _mm256_loadu_ps(&layer->layer_grad->biases->data[o]);
            output_vec = _mm256_add_ps(output_vec, bias_vec);
            _mm256_storeu_ps(&output_row[o], output_vec);
        }
        for (; o < layer->output_size; ++o) {
            output_row[o] += layer->layer_grad->biases->data[o];
        }
    }


    LinearOutput* result = (LinearOutput*)malloc(sizeof(LinearOutput));
    if (!result) {
        tensor_free(output);
        return NULL;
    }

    result->output = output;
    result->layer = layer;

    return result;
}

void linear_output_free(LinearOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        free(result);
    }
}

LinearBackwardOutput* linear_backward(Linear* layer, LinearOutput* forward_result,
                                     Tensor* output_grad) {
    if (!layer || !forward_result || !output_grad || !layer->input_cache) return NULL;

    if (output_grad->shape[0] != forward_result->output->shape[0] ||
        output_grad->shape[1] != layer->output_size) {
        printf("Error: Output gradient dimensions don't match\n");
        return NULL;
    }

    size_t batch_size = (size_t)output_grad->shape[0];

    int input_grad_shape[2] = {(int)batch_size, layer->input_size};
    Tensor* input_grad = tensor_create(input_grad_shape, 2);
    if (!input_grad) return NULL;

    if (layer->layer_grad->weight_grad) {
        const int M = layer->input_size;
        const int K = batch_size;
        const int N = layer->output_size;

        memset(layer->layer_grad->weight_grad->data, 0, layer->layer_grad->weight_grad->size * sizeof(float));

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < K; ++k) {
            const float* input_row = &layer->input_cache->data[k * M];
            const float* grad_row = &output_grad->data[k * N];

            for (int m = 0; m < M; ++m) {
                const float input_val = input_row[m];
                float* weight_row = &layer->layer_grad->weight_grad->data[m * N];

                __m256 input_broadcast = _mm256_set1_ps(input_val);

                int n = 0;
                for (; n <= N - 8; n += 8) {
                    __m256 weight_vec = _mm256_loadu_ps(&weight_row[n]);
                    __m256 grad_vec = _mm256_loadu_ps(&grad_row[n]);
                    weight_vec = _mm256_fmadd_ps(input_broadcast, grad_vec, weight_vec);
                    _mm256_storeu_ps(&weight_row[n], weight_vec);
                }

                for (; n < N; ++n) {
                    weight_row[n] += input_val * grad_row[n];
                }
            }
        }
    }

    Tensor* bias_grad_sum = tensor_sum_axis(output_grad, 0);
    if (bias_grad_sum) {
        memcpy(layer->layer_grad->bias_grad->data, bias_grad_sum->data,
               layer->output_size * sizeof(float));
        tensor_free(bias_grad_sum);
    }
    const int M = batch_size;
    const int K = layer->output_size;
    const int N = layer->input_size;

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const float* grad_row = &output_grad->data[m * K];
        float* input_row = &input_grad->data[m * N];

        for (int n = 0; n < N; ++n) {
            const float* weight_row = &layer->layer_grad->weights->data[n * K];

            float sum = 0.0f;
            __m256 sum_vec = _mm256_setzero_ps();

            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 grad_vec = _mm256_loadu_ps(&grad_row[k]);
                __m256 weight_vec = _mm256_loadu_ps(&weight_row[k]);
                sum_vec = _mm256_fmadd_ps(grad_vec, weight_vec, sum_vec);
            }

            float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            sum = temp[0] + temp[1] + temp[2] + temp[3] +
                  temp[4] + temp[5] + temp[6] + temp[7];

            for (; k < K; ++k) {
                sum += grad_row[k] * weight_row[k];
            }

            input_row[n] = sum;
        }
    }

    LinearBackwardOutput* result = (LinearBackwardOutput*)malloc(sizeof(LinearBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void linear_backward_output_free(LinearBackwardOutput* result) {
    if (result) {
        if (result->input_grad) tensor_free(result->input_grad);
        free(result);
    }
}

