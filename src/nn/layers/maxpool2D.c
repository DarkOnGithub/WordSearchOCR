#include "nn/layers/maxpool2D.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stdbool.h>
#include <immintrin.h>

static inline void find_max(const float* values, int count, float* max_val, int* max_idx) {
    *max_val = -FLT_MAX;
    *max_idx = -1;

    if (count <= 0) return;

    *max_val = values[0];
    *max_idx = 0;

    int i = 1;
    __m256 current_max_vec = _mm256_set1_ps(*max_val);

    for (; i <= count - 8; i += 8) {
        __m256 vals = _mm256_loadu_ps(&values[i]);
        __m256 cmp_result = _mm256_cmp_ps(vals, current_max_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_ps(cmp_result);

        if (mask != 0) {
            for (int j = 0; j < 8; j++) {
                if (values[i + j] > *max_val) {
                    *max_val = values[i + j];
                    *max_idx = i + j;
                }
            }
            current_max_vec = _mm256_set1_ps(*max_val);
        }
    }

    for (; i < count; i++) {
        if (values[i] > *max_val) {
            *max_val = values[i];
            *max_idx = i;
        }
    }
}

static inline void find_kernel_max(const float* input_data, int height, int width,
                                       int ih_start, int iw_start, int kernel_h, int kernel_w,
                                       int dilation_h, int dilation_w, int batch_offset, int channel_offset,
                                       int height_stride, float* max_val, int* max_linear_idx) {
    *max_val = -FLT_MAX;
    *max_linear_idx = -1;

    // For dilation > 1, use scalar version
    if (dilation_h > 1 || dilation_w > 1) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = ih_start + kh * dilation_h;
                int iw = iw_start + kw * dilation_w;

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
        return;
    }

    // For dilation = 1, collect values and max finding
    int max_possible = kernel_h * kernel_w;
    if (max_possible <= 32) {
        // Small kernel, use simple scalar version
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = ih_start + kh;
                int iw = iw_start + kw;

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
    } else {
        // Larger kernel, collect values and use SIMD
        float* kernel_values = (float*)malloc(sizeof(float) * max_possible);
        int* kernel_indices = (int*)malloc(sizeof(int) * max_possible);
        if (!kernel_values || !kernel_indices) {
            free(kernel_values);
            free(kernel_indices);
            find_kernel_max(input_data, height, width, ih_start, iw_start,
                               kernel_h, kernel_w, 1, 1, batch_offset, channel_offset,
                               height_stride, max_val, max_linear_idx);
            return;
        }

        int valid_count = 0;
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = ih_start + kh;
                int iw = iw_start + kw;

                if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                    int input_idx = batch_offset + channel_offset + ih * height_stride + iw;
                    kernel_values[valid_count] = input_data[input_idx];
                    kernel_indices[valid_count] = input_idx;
                    valid_count++;
                }
            }
        }

        if (valid_count > 0) {
            int max_pos;
            find_max(kernel_values, valid_count, max_val, &max_pos);
            *max_linear_idx = kernel_indices[max_pos];
        }

        free(kernel_values);
        free(kernel_indices);
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

    for (int b = 0; b < batch_size; ++b) {
        int batch_offset = b * batch_stride;
        for (int c = 0; c < channels; ++c) {
            int channel_offset = c * channel_stride;
            for (int oh = 0; oh < output_h; ++oh) {
                for (int ow = 0; ow < output_w; ++ow) {
                    int ih_start = oh * layer->stride_h - layer->padding_h;
                    int iw_start = ow * layer->stride_w - layer->padding_w;

                    float max_val;
                    int max_idx;
                    find_kernel_max(input->data, height, width, ih_start, iw_start,
                                       layer->kernel_size_h, layer->kernel_size_w,
                                       layer->dilation_h, layer->dilation_w,
                                       batch_offset, channel_offset, width,
                                       &max_val, &max_idx);

                    int output_idx = b * output_batch_stride + c * output_channel_stride +
                                   oh * output_w + ow;
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
        // Use indices for direct gradient routing
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

                        // Route gradient to that position
                        input_grad->data[input_linear_idx] += grad_val;
                    }
                }
            }
        }
    } else {
        // No indices available, need to recompute maximum positions
        // This is less efficient but works when indices aren't stored

        // Recompute maximum positions by iterating through the input again
        Tensor* input = forward_result->input;
        int in_batch = input->shape[0];
        int in_channels = input->shape[1];
        int in_height = input->shape[2];
        int in_width = input->shape[3];

        int in_channel_stride = in_height * in_width;
        int in_batch_stride = in_channels * in_channel_stride;
        int out_channel_stride = out_height * out_width;
        int out_batch_stride = out_channels * out_channel_stride;

        for (int b = 0; b < in_batch; ++b) {
            int batch_offset = b * in_batch_stride;
            for (int c = 0; c < in_channels; ++c) {
                int channel_offset = c * in_channel_stride;
                for (int oh = 0; oh < out_height; ++oh) {
                    for (int ow = 0; ow < out_width; ++ow) {
                        int output_idx = b * out_batch_stride + c * out_channel_stride +
                                       oh * out_width + ow;
                        float grad_val = output_grad->data[output_idx];

                        int ih_start = oh * layer->stride_h - layer->padding_h;
                        int iw_start = ow * layer->stride_w - layer->padding_w;

                        float max_val;
                        int max_linear_idx;
                        find_kernel_max(input->data, in_height, in_width, ih_start, iw_start,
                                           layer->kernel_size_h, layer->kernel_size_w,
                                           layer->dilation_h, layer->dilation_w,
                                           batch_offset, channel_offset, in_width,
                                           &max_val, &max_linear_idx);

                        if (max_linear_idx != -1) {
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
