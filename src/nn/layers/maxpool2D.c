#include "nn/layers/maxpool2D.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <immintrin.h>
#include <omp.h>

static __attribute__((always_inline)) inline void find_kernel_max(const float* input_data, int height, int width,
                                       int ih_start, int iw_start, int kernel_h, int kernel_w,
                                       int dilation_h, int dilation_w, int batch_offset, int channel_offset,
                                       int height_stride, float* max_val, int* max_linear_idx) {
    *max_val = -FLT_MAX;
    *max_linear_idx = -1;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = ih_start + kh * dilation_h;
            int iw = iw_start + kw * dilation_w;

            // Check if the pixel is within the valid input bounds
            if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                int input_idx = batch_offset + channel_offset + ih * height_stride + iw;
                float val = input_data[input_idx];

                if (val > *max_val) {
                    *max_val = val;
                    *max_linear_idx = input_idx;
                }
            }
        }
    }
}

MaxPool2D* maxpool2d_create(int kernel_size_h, int kernel_size_w,
                          int stride_h, int stride_w,
                          int padding_h, int padding_w,
                          int dilation_h, int dilation_w,
                          bool return_indices, bool ceil_mode) {
    MaxPool2D* layer = (MaxPool2D*)malloc(sizeof(MaxPool2D));
    if (!layer) return NULL;

    layer->kernel_size_h = kernel_size_h;
    layer->kernel_size_w = kernel_size_w;
    layer->stride_h = stride_h;
    layer->stride_w = stride_w;
    layer->padding_h = padding_h;
    layer->padding_w = padding_w;
    layer->dilation_h = dilation_h;
    layer->dilation_w = dilation_w;
    layer->return_indices = return_indices;
    layer->ceil_mode = ceil_mode;

    return layer;
}

MaxPool2D* maxpool2d_create_square(int kernel_size, int stride, int padding,
                                 bool return_indices, bool ceil_mode) {
    return maxpool2d_create(kernel_size, kernel_size, stride, stride,
                           padding, padding, 1, 1, return_indices, ceil_mode);
}

MaxPool2D* maxpool2d_create_simple(int kernel_size, int stride) {
    return maxpool2d_create_square(kernel_size, stride, 0, false, false);
}

void maxpool2d_free(MaxPool2D* layer) {
    if (layer) {
        free(layer);
    }
}

void maxpool2d_get_output_dims(MaxPool2D* layer, int input_h, int input_w,
                              int* output_h, int* output_w) {
    if (layer->ceil_mode) {
        // Use ceil mode: allow sliding windows to go off-bounds if they start within padding
        *output_h = (int)ceil((double)(input_h + 2 * layer->padding_h -
                                       layer->dilation_h * (layer->kernel_size_h - 1) - 1) /
                              layer->stride_h) + 1;
        *output_w = (int)ceil((double)(input_w + 2 * layer->padding_w -
                                       layer->dilation_w * (layer->kernel_size_w - 1) - 1) /
                              layer->stride_w) + 1;
    } else {
        // Standard floor mode
        *output_h = (int)floor((double)(input_h + 2 * layer->padding_h -
                                        layer->dilation_h * (layer->kernel_size_h - 1) - 1) /
                               layer->stride_h) + 1;
        *output_w = (int)floor((double)(input_w + 2 * layer->padding_w -
                                        layer->dilation_w * (layer->kernel_size_w - 1) - 1) /
                               layer->stride_w) + 1;
    }
}

MaxPool2DOutput* maxpool2d_forward(MaxPool2D* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    int output_h, output_w;
    maxpool2d_get_output_dims(layer, input->shape[2], input->shape[3], &output_h, &output_w);

    int output_shape[4] = {input->shape[0], input->shape[1], output_h, output_w};
    Tensor* output = tensor_create(output_shape, 4);
    if (!output) return NULL;

    Tensor* indices = NULL;
    if (layer->return_indices) {
        indices = tensor_create(output_shape, 4);
        if (!indices) {
            tensor_free(output);
            return NULL;
        }
    }

    int batch_size = input->shape[0];
    int channels = input->shape[1];
    int height = input->shape[2];
    int width = input->shape[3];

    int channel_stride = height * width;
    int batch_stride = channels * channel_stride;
    int output_channel_stride = output_h * output_w;
    int output_batch_stride = channels * output_channel_stride;

    // Precompute constants for better performance
    const int dilation_h = layer->dilation_h;
    const int dilation_w = layer->dilation_w;
    const int kernel_h = layer->kernel_size_h;
    const int kernel_w = layer->kernel_size_w;
    const int stride_h = layer->stride_h;
    const int stride_w = layer->stride_w;
    const int padding_h = layer->padding_h;
    const int padding_w = layer->padding_w;

    // Optimized loop with better memory access patterns and OpenMP parallelization
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            const int batch_offset = b * batch_stride;
            const int channel_offset = c * channel_stride;
            const int output_base_idx = b * output_batch_stride + c * output_channel_stride;

            for (int oh = 0; oh < output_h; ++oh) {
                const int ih_start = oh * stride_h - padding_h;
                const int output_row_base = output_base_idx + oh * output_w;

                // Process output positions in row with some unrolling for small widths
                int ow = 0;
                for (; ow <= output_w - 4; ow += 4) {
                    // Unroll 4 iterations for better ILP
                    for (int k = 0; k < 4; ++k) {
                        const int ow_k = ow + k;
                        const int iw_start = ow_k * stride_w - padding_w;

                        float max_val;
                        int max_idx;
                        find_kernel_max(input->data, height, width, ih_start, iw_start,
                                       kernel_h, kernel_w, dilation_h, dilation_w,
                                       batch_offset, channel_offset, width,
                                       &max_val, &max_idx);

                        const int output_idx = output_row_base + ow_k;
                        output->data[output_idx] = max_val;

                        if (layer->return_indices && indices) {
                            indices->data[output_idx] = (float)max_idx;
                        }
                    }
                }

                // Handle remaining elements
                for (; ow < output_w; ++ow) {
                    const int iw_start = ow * stride_w - padding_w;

                    float max_val;
                    int max_idx;
                    find_kernel_max(input->data, height, width, ih_start, iw_start,
                                   kernel_h, kernel_w, dilation_h, dilation_w,
                                   batch_offset, channel_offset, width,
                                   &max_val, &max_idx);

                    const int output_idx = output_row_base + ow;
                    output->data[output_idx] = max_val;

                    if (layer->return_indices && indices) {
                        indices->data[output_idx] = (float)max_idx;
                    }
                }
            }
        }
    }

    MaxPool2DOutput* result = (MaxPool2DOutput*)malloc(sizeof(MaxPool2DOutput));
    if (!result) {
        tensor_free(output);
        if (indices) tensor_free(indices);
        return NULL;
    }

    result->output = output;
    result->indices = indices;
    result->input = input;

    return result;
}

void maxpool2d_output_free(MaxPool2DOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        if (result->indices) tensor_free(result->indices);
        free(result);
    }
}

MaxPool2DBackwardOutput* maxpool2d_backward(MaxPool2D* layer, MaxPool2DOutput* forward_result,
                                          Tensor* output_grad) {
    if (!layer || !forward_result || !forward_result->output || !forward_result->input || !output_grad) {
        return NULL;
    }
    if (output_grad->shape[0] != forward_result->output->shape[0] ||
        output_grad->shape[1] != forward_result->output->shape[1] ||
        output_grad->shape[2] != forward_result->output->shape[2] ||
        output_grad->shape[3] != forward_result->output->shape[3]) {
        printf("Error: Output gradient dimensions don't match forward output dimensions\n");
        return NULL;
    }

    Tensor* input_grad = tensor_create_zero(forward_result->input->shape, forward_result->input->ndim);
    if (!input_grad) return NULL;

    int out_batch = forward_result->output->shape[0];
    int out_channels = forward_result->output->shape[1];
    int out_height = forward_result->output->shape[2];
    int out_width = forward_result->output->shape[3];

    if (forward_result->indices && layer->return_indices) {
        // Fast path: Use indices for direct gradient routing with OpenMP parallelization
        #pragma omp parallel for collapse(4)
        for (int b = 0; b < out_batch; ++b) {
            for (int c = 0; c < out_channels; ++c) {
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int output_idx = b * (out_channels * out_height * out_width) +
                                       c * (out_height * out_width) +
                                       oh * out_width + ow;
                        float grad_val = output_grad->data[output_idx];

                        // Get the input index where the maximum came from
                        int input_linear_idx = (int)forward_result->indices->data[output_idx];

                        // MUST be atomic to prevent race conditions
                        #pragma omp atomic
                        input_grad->data[input_linear_idx] += grad_val;
                    }
                }
            }
        }
    } else {
        // Slow path: No indices available, need to recompute maximum positions
        // Optimized version with same loop structure as forward pass

        Tensor* input = forward_result->input;
        const int in_batch = input->shape[0];
        const int in_channels = input->shape[1];
        const int in_height = input->shape[2];
        const int in_width = input->shape[3];

        const int in_channel_stride = in_height * in_width;
        const int in_batch_stride = in_channels * in_channel_stride;
        const int out_channel_stride = out_height * out_width;
        const int out_batch_stride = out_channels * out_channel_stride;

        // Precompute constants
        const int dilation_h = layer->dilation_h;
        const int dilation_w = layer->dilation_w;
        const int kernel_h = layer->kernel_size_h;
        const int kernel_w = layer->kernel_size_w;
        const int stride_h = layer->stride_h;
        const int stride_w = layer->stride_w;
        const int padding_h = layer->padding_h;
        const int padding_w = layer->padding_w;

        #pragma omp parallel for collapse(2)
        for (int b = 0; b < in_batch; ++b) {
            for (int c = 0; c < in_channels; ++c) {
                const int batch_offset = b * in_batch_stride;
                const int channel_offset = c * in_channel_stride;
                const int output_base_idx = b * out_batch_stride + c * out_channel_stride;

                for (int oh = 0; oh < out_height; ++oh) {
                    const int ih_start = oh * stride_h - padding_h;
                    const int output_row_base = output_base_idx + oh * out_width;

                    // Process output positions in row with unrolling
                    int ow = 0;
                    for (; ow <= out_width - 4; ow += 4) {
                        for (int k = 0; k < 4; ++k) {
                            const int ow_k = ow + k;
                            const int iw_start = ow_k * stride_w - padding_w;
                            const int output_idx = output_row_base + ow_k;
                            const float grad_val = output_grad->data[output_idx];

                            float max_val;
                            int max_linear_idx;
                            find_kernel_max(input->data, in_height, in_width, ih_start, iw_start,
                                           kernel_h, kernel_w, dilation_h, dilation_w,
                                           batch_offset, channel_offset, in_width,
                                           &max_val, &max_linear_idx);

                            if (max_linear_idx != -1) {
                                // MUST be atomic here too
                                #pragma omp atomic
                                input_grad->data[max_linear_idx] += grad_val;
                            }
                        }
                    }

                    // Handle remaining elements
                    for (; ow < out_width; ++ow) {
                        const int iw_start = ow * stride_w - padding_w;
                        const int output_idx = output_row_base + ow;
                        const float grad_val = output_grad->data[output_idx];

                        float max_val;
                        int max_linear_idx;
                        find_kernel_max(input->data, in_height, in_width, ih_start, iw_start,
                                       kernel_h, kernel_w, dilation_h, dilation_w,
                                       batch_offset, channel_offset, in_width,
                                       &max_val, &max_linear_idx);

                        if (max_linear_idx != -1) {
                            // MUST be atomic here too
                            #pragma omp atomic
                            input_grad->data[max_linear_idx] += grad_val;
                        }
                    }
                }
            }
        }
    }

    MaxPool2DBackwardOutput* result = (MaxPool2DBackwardOutput*)malloc(sizeof(MaxPool2DBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void maxpool2d_backward_output_free(MaxPool2DBackwardOutput* result) {
    if (result) {
        if (result->input_grad) tensor_free(result->input_grad);
        free(result);
    }
}
