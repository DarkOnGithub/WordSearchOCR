#include "nn/layers/adaptive_avg_pool2D.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

AdaptiveAvgPool2D* adaptive_avg_pool2d_create(int output_height, int output_width) {
    if (output_height <= 0 || output_width <= 0) {
        printf("Error: Output dimensions must be positive\n");
        return NULL;
    }

    AdaptiveAvgPool2D* layer = (AdaptiveAvgPool2D*)malloc(sizeof(AdaptiveAvgPool2D));
    if (!layer) return NULL;

    layer->output_height = output_height;
    layer->output_width = output_width;
    layer->input_cache = NULL;

    return layer;
}

static void compute_bin_boundaries(int input_size, int output_size,
                                   int* bin_starts, int* bin_sizes) {
    for (int i = 0; i < output_size; i++) {
        float start_float = (float)i * input_size / output_size;
        float end_float = (float)(i + 1) * input_size / output_size;

        bin_starts[i] = (int)floorf(start_float);
        int bin_end = (int)ceilf(end_float);
        bin_sizes[i] = bin_end - bin_starts[i];
    }
}


AdaptiveAvgPool2DOutput* adaptive_avg_pool2d_forward(AdaptiveAvgPool2D* layer, Tensor* input) {
    if (!layer || !input) {
        printf("Error: Invalid input parameters\n");
        return NULL;
    }

    if (input->ndim != 4) {
        printf("Error: Input tensor must be 4D [N, C, H, W]\n");
        return NULL;
    }

    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int out_H = layer->output_height;
    int out_W = layer->output_width;

    int output_shape[4] = {N, C, out_H, out_W};
    Tensor* output = tensor_create_zero(output_shape, 4);
    if (!output) return NULL;

    int* h_bin_starts = (int*)malloc(out_H * sizeof(int));
    int* h_bin_sizes = (int*)malloc(out_H * sizeof(int));
    int* w_bin_starts = (int*)malloc(out_W * sizeof(int));
    int* w_bin_sizes = (int*)malloc(out_W * sizeof(int));

    if (!h_bin_starts || !h_bin_sizes || !w_bin_starts || !w_bin_sizes) {
        free(h_bin_starts); free(h_bin_sizes); free(w_bin_starts); free(w_bin_sizes);
        tensor_free(output);
        return NULL;
    }

    compute_bin_boundaries(H, out_H, h_bin_starts, h_bin_sizes);
    compute_bin_boundaries(W, out_W, w_bin_starts, w_bin_sizes);

    if (layer->input_cache) {
        tensor_free(layer->input_cache);
    }
    layer->input_cache = tensor_from_data(input->data, input->shape, input->ndim);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    int h_start = h_bin_starts[oh];
                    int h_size = h_bin_sizes[oh];
                    int w_start = w_bin_starts[ow];
                    int w_size = w_bin_sizes[ow];

                    float sum = 0.0f;
                    int count = 0;

                    for (int h = h_start; h < h_start + h_size && h < H; h++) {
                        for (int w = w_start; w < w_start + w_size && w < W; w++) {
                            int input_idx = ((n * C + c) * H + h) * W + w;
                            sum += input->data[input_idx];
                            count++;
                        }
                    }

                    float avg = (count > 0) ? sum / count : 0.0f;

                    int output_idx = ((n * C + c) * out_H + oh) * out_W + ow;
                    output->data[output_idx] = avg;
                }
            }
        }
    }

    free(h_bin_starts); free(h_bin_sizes); free(w_bin_starts); free(w_bin_sizes);

    AdaptiveAvgPool2DOutput* result = (AdaptiveAvgPool2DOutput*)malloc(sizeof(AdaptiveAvgPool2DOutput));
    if (!result) {
        tensor_free(output);
        return NULL;
    }

    result->output = output;
    result->layer = layer;

    return result;
}

AdaptiveAvgPool2DBackwardOutput* adaptive_avg_pool2d_backward(AdaptiveAvgPool2D* layer,
                                                             AdaptiveAvgPool2DOutput* forward_result,
                                                             Tensor* output_grad) {
    if (!layer || !forward_result || !output_grad || !layer->input_cache) {
        printf("Error: Invalid backward pass parameters\n");
        return NULL;
    }

    Tensor* input = layer->input_cache;
    int N = input->shape[0];
    int C = input->shape[1];
    int H = input->shape[2];
    int W = input->shape[3];
    int out_H = layer->output_height;
    int out_W = layer->output_width;

    Tensor* input_grad = tensor_create_zero(input->shape, input->ndim);
    if (!input_grad) return NULL;

    int* h_bin_starts = (int*)malloc(out_H * sizeof(int));
    int* h_bin_sizes = (int*)malloc(out_H * sizeof(int));
    int* w_bin_starts = (int*)malloc(out_W * sizeof(int));
    int* w_bin_sizes = (int*)malloc(out_W * sizeof(int));

    if (!h_bin_starts || !h_bin_sizes || !w_bin_starts || !w_bin_sizes) {
        free(h_bin_starts); free(h_bin_sizes); free(w_bin_starts); free(w_bin_sizes);
        tensor_free(input_grad);
        return NULL;
    }

    compute_bin_boundaries(H, out_H, h_bin_starts, h_bin_sizes);
    compute_bin_boundaries(W, out_W, w_bin_starts, w_bin_sizes);

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_H; oh++) {
                for (int ow = 0; ow < out_W; ow++) {
                    int h_start = h_bin_starts[oh];
                    int h_size = h_bin_sizes[oh];
                    int w_start = w_bin_starts[ow];
                    int w_size = w_bin_sizes[ow];

                    int count = 0;
                    for (int h = h_start; h < h_start + h_size && h < H; h++) {
                        for (int w = w_start; w < w_start + w_size && w < W; w++) {
                            count++;
                        }
                    }

                    if (count == 0) continue;

                    float grad_value = output_grad->data[((n * C + c) * out_H + oh) * out_W + ow] / count;

                    for (int h = h_start; h < h_start + h_size && h < H; h++) {
                        for (int w = w_start; w < w_start + w_size && w < W; w++) {
                            int input_idx = ((n * C + c) * H + h) * W + w;
                            input_grad->data[input_idx] += grad_value;
                        }
                    }
                }
            }
        }
    }

    free(h_bin_starts); free(h_bin_sizes); free(w_bin_starts); free(w_bin_sizes);

    AdaptiveAvgPool2DBackwardOutput* result = (AdaptiveAvgPool2DBackwardOutput*)malloc(sizeof(AdaptiveAvgPool2DBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void adaptive_avg_pool2d_free(AdaptiveAvgPool2D* layer) {
    if (!layer) return;

    if (layer->input_cache) {
        tensor_free(layer->input_cache);
    }
    free(layer);
}

void adaptive_avg_pool2d_output_free(AdaptiveAvgPool2DOutput* result) {
    if (!result) return;

    if (result->output) {
        tensor_free(result->output);
    }
    free(result);
}

void adaptive_avg_pool2d_backward_output_free(AdaptiveAvgPool2DBackwardOutput* result) {
    if (!result) return;

    if (result->input_grad) {
        tensor_free(result->input_grad);
    }
    free(result);
}
