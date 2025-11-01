#include "nn/layers/conv2D.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#ifdef _OPENMP
#include <omp.h>
#endif

// Highly optimized 3x3 convolution for stride=1, padding=1 (interior only)
// Adds into output 8 pixels starting at out_w_start (requires 1 <= out_w_start <= output_width-9 and 1 <= out_h <= output_height-2)
static inline void conv3x3_8pixels_avx2(
    float* __restrict output, const float* __restrict input, const float* __restrict kernel,
    int input_width, int input_height, int output_width, int out_h, int out_w_start
) {
    (void)input_height; // unused in interior kernel

    // Broadcast kernel
    __m256 k0 = _mm256_set1_ps(kernel[0]);
    __m256 k1 = _mm256_set1_ps(kernel[1]);
    __m256 k2 = _mm256_set1_ps(kernel[2]);
    __m256 k3 = _mm256_set1_ps(kernel[3]);
    __m256 k4 = _mm256_set1_ps(kernel[4]);
    __m256 k5 = _mm256_set1_ps(kernel[5]);
    __m256 k6 = _mm256_set1_ps(kernel[6]);
    __m256 k7 = _mm256_set1_ps(kernel[7]);
    __m256 k8 = _mm256_set1_ps(kernel[8]);

    const int base_out = out_h * output_width + out_w_start;
    const int in_row0 = (out_h - 1) * input_width + (out_w_start - 1);
    const int in_row1 = (out_h     ) * input_width + (out_w_start - 1);
    const int in_row2 = (out_h + 1) * input_width + (out_w_start - 1);

    // Load existing output to accumulate into
    __m256 outv = _mm256_loadu_ps(&output[base_out]);

    // Top row
    __m256 r0l = _mm256_loadu_ps(&input[in_row0 + 0]);
    __m256 r0c = _mm256_loadu_ps(&input[in_row0 + 1]);
    __m256 r0r = _mm256_loadu_ps(&input[in_row0 + 2]);

    // Middle row
    __m256 r1l = _mm256_loadu_ps(&input[in_row1 + 0]);
    __m256 r1c = _mm256_loadu_ps(&input[in_row1 + 1]);
    __m256 r1r = _mm256_loadu_ps(&input[in_row1 + 2]);

    // Bottom row
    __m256 r2l = _mm256_loadu_ps(&input[in_row2 + 0]);
    __m256 r2c = _mm256_loadu_ps(&input[in_row2 + 1]);
    __m256 r2r = _mm256_loadu_ps(&input[in_row2 + 2]);

    // FMA accumulate
    outv = _mm256_fmadd_ps(r0l, k0, outv);
    outv = _mm256_fmadd_ps(r0c, k1, outv);
    outv = _mm256_fmadd_ps(r0r, k2, outv);
    outv = _mm256_fmadd_ps(r1l, k3, outv);
    outv = _mm256_fmadd_ps(r1c, k4, outv);
    outv = _mm256_fmadd_ps(r1r, k5, outv);
    outv = _mm256_fmadd_ps(r2l, k6, outv);
    outv = _mm256_fmadd_ps(r2c, k7, outv);
    outv = _mm256_fmadd_ps(r2r, k8, outv);

    _mm256_storeu_ps(&output[base_out], outv);
}

// Optimized 3x3 convolution for single channel (stride=1, padding=1)
// Accumulates into output (does not zero it)
static inline void conv3x3_single_channel_optimized(
    float* __restrict output, const float* __restrict input, const float* __restrict kernel,
    int output_width, int output_height, int input_width, int input_height
) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
        if (out_h > 0 && out_h < output_height - 1) {
            // Left border (w=0)
            {
                const int in_h_start = out_h;
                const int in_w_start = 0;
                float sum = 0.0f;
                // Unrolled with boundary checks
                int kh = -1, kw = -1;
                if (in_h_start + kh >= 0 && (in_w_start + kw) >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[0];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[1];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[2];
                }
                kh = 0; kw = -1; if ((in_w_start + kw) >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[3];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[4];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[5];
                }
                kh = 1; kw = -1; if ((in_w_start + kw) >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[6];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[7];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[8];
                }
                output[out_h * output_width + 0] += sum;
            }

            // Vectorized interior (8 at a time), requires 1 <= w < W-9
            int out_w = 1;
            for (; out_w < output_width - 9; out_w += 8) {
                conv3x3_8pixels_avx2(output, input, kernel, input_width, input_height, output_width, out_h, out_w);
            }

            // Tail and right border
            for (; out_w < output_width; ++out_w) {
                const int in_h_start = out_h;
                const int in_w_start = out_w;
                float sum = 0.0f;

                int kh = -1, kw = -1;
                if (in_w_start + kw >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[0];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[1];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[2];
                }

                kh = 0; kw = -1; if (in_w_start + kw >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[3];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[4];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[5];
                }

                kh = 1; kw = -1; if (in_w_start + kw >= 0) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[6];
                }
                kw = 0; sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[7];
                kw = 1; if (in_w_start + kw < input_width) {
                    sum += input[(in_h_start + kh) * input_width + (in_w_start + kw)] * kernel[8];
                }

                output[out_h * output_width + out_w] += sum;
            }
        } else {
            // Top or bottom border rows: scalar with full checks
            for (int out_w = 0; out_w < output_width; ++out_w) {
                const int in_h_start = out_h;
                const int in_w_start = out_w;
                float sum = 0.0f;

                for (int kh = -1; kh <= 1; ++kh) {
                    int ih = in_h_start + kh;
                    if (ih < 0 || ih >= input_height) continue;
                    for (int kw = -1; kw <= 1; ++kw) {
                        int iw = in_w_start + kw;
                        if (iw < 0 || iw >= input_width) continue;
                        sum += input[ih * input_width + iw] * kernel[(kh + 1) * 3 + (kw + 1)];
                    }
                }
                output[out_h * output_width + out_w] += sum;
            }
        }
    }
}

// Generic SIMD convolution for other kernel sizes (fallback)
static inline void conv_single_channel_simd(
    float* output, const float* input, const float* kernel,
    int output_width, int output_height, int input_width, int input_height,
    int kernel_size, int stride, int padding
) {
    for (int out_h = 0; out_h < output_height; ++out_h) {
        for (int out_w = 0; out_w < output_width; ++out_w) {
            const int in_h_start = out_h * stride - padding;
            const int in_w_start = out_w * stride - padding;

            float sum = 0.0f;
            for (int kh = 0; kh < kernel_size; ++kh) {
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int in_h = in_h_start + kh;
                    int in_w = in_w_start + kw;

                    if (in_h >= 0 && in_h < input_height && in_w >= 0 && in_w < input_width) {
                        sum += input[in_h * input_width + in_w] * kernel[kh * kernel_size + kw];
                    }
                }
            }
            output[out_h * output_width + out_w] += sum;
        }
    }
}


// He initialization for weights (better for ReLU)
static void he_init(float* weights, int count, int fan_in) {
    float scale = sqrtf(2.0f / fan_in);

    for (int i = 0; i < count; ++i) {
        weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

Conv2D* conv2D_create(int input_channels, int output_channels, int kernel_size,
                      int stride, int padding) {
    Conv2D* conv = (Conv2D*)malloc(sizeof(Conv2D));
    if (!conv) return NULL;

    conv->kernel_size = kernel_size;
    conv->stride = stride;
    conv->padding = padding;

    // Create weight tensor: [output_channels, input_channels, kernel_size, kernel_size]
    int weight_shape[] = {output_channels, input_channels, kernel_size, kernel_size};
    conv->weight = tensor_create(weight_shape, 4);
    if (!conv->weight) {
        free(conv);
        return NULL;
    }

    // Create bias tensor: [output_channels, 1, 1, 1]
    int bias_shape[] = {output_channels, 1, 1, 1};
    conv->bias = tensor_create(bias_shape, 4);
    if (!conv->bias) {
        tensor_free(conv->weight);
        free(conv);
        return NULL;
    }

    // Create gradient tensors with same shapes
    conv->weight_grad = tensor_create_zero(weight_shape, 4);
    if (!conv->weight_grad) {
        tensor_free(conv->weight);
        tensor_free(conv->bias);
        free(conv);
        return NULL;
    }

    conv->bias_grad = tensor_create_zero(bias_shape, 4);
    if (!conv->bias_grad) {
        tensor_free(conv->weight);
        tensor_free(conv->bias);
        tensor_free(conv->weight_grad);
        free(conv);
        return NULL;
    }

    // Initialize weights using He initialization (good for ReLU activations)
    const int fan_in = input_channels * kernel_size * kernel_size;
    const int weight_count = output_channels * fan_in;
    he_init(conv->weight->data, weight_count, fan_in);

    // Initialize bias to zeros
    memset(conv->bias->data, 0, conv->bias->size * sizeof(float));

    return conv;
}

void conv2D_free(Conv2D* conv) {
    if (conv) {
        if (conv->weight) tensor_free(conv->weight);
        if (conv->bias) tensor_free(conv->bias);
        if (conv->weight_grad) tensor_free(conv->weight_grad);
        if (conv->bias_grad) tensor_free(conv->bias_grad);
        free(conv);
    }
}

Tensor* conv2D_forward(Conv2D* conv, Tensor* input) {
    if (!conv || !input) return NULL;

    const int batch_size = input->shape[0];
    const int input_channels = input->shape[1];
    const int input_height = input->shape[2];
    const int input_width = input->shape[3];

    // Calculate output dimensions
    const int output_height = (input_height + 2 * conv->padding - conv->kernel_size) / conv->stride + 1;
    const int output_width = (input_width + 2 * conv->padding - conv->kernel_size) / conv->stride + 1;
    const int output_channels = conv->weight->shape[0];

    if (output_height <= 0 || output_width <= 0) {
        printf("Error: Invalid output dimensions: height=%d, width=%d\n", output_height, output_width);
        return NULL;
    }

    // Create output tensor: [batch_size, output_channels, output_height, output_width]
    int output_shape[] = {batch_size, output_channels, output_height, output_width};
    Tensor* output = tensor_create(output_shape, 4);
    if (!output) return NULL;

    // Process each batch and output channel (parallelizable)
    #pragma omp parallel for collapse(2) schedule(static) if (batch_size * output_channels > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            float* output_channel = &output->data[b * output_channels * output_width * output_height +
                                                 oc * output_width * output_height];

            // Zero the output channel before accumulating
            memset(output_channel, 0, output_width * output_height * sizeof(float));

            // For each input channel
            for (int ic = 0; ic < input_channels; ++ic) {
                const float* kernel = &conv->weight->data[oc * input_channels * conv->kernel_size * conv->kernel_size +
                                                          ic * conv->kernel_size * conv->kernel_size];
                const float* input_channel = &input->data[b * input_channels * input_width * input_height +
                                                         ic * input_width * input_height];

                // Use optimized 3x3 convolution for the common case (stride=1, padding=1, kernel_size=3)
                if (conv->kernel_size == 3 && conv->stride == 1 && conv->padding == 1) {
                    conv3x3_single_channel_optimized(
                        output_channel, input_channel, kernel,
                        output_width, output_height, input_width, input_height
                    );
                } else {
                    // Fallback to generic SIMD convolution
                    conv_single_channel_simd(
                        output_channel, input_channel, kernel,
                        output_width, output_height, input_width, input_height,
                        conv->kernel_size, conv->stride, conv->padding
                    );
                }
            }
        }
    }

    // Add bias (broadcasting across spatial dimensions)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            const float bias_val = conv->bias->data[oc];
            float* output_channel = &output->data[b * output_channels * output_width * output_height +
                                                  oc * output_width * output_height];

            // Add bias to each position in the output channel (SIMD-optimized)
            __m256 bias_vec = _mm256_set1_ps(bias_val);
            const size_t out_size = (size_t)output_width * output_height;

            for (size_t i = 0; i < out_size; i += 8) {
                if (i + 8 <= out_size) {
                    __m256 current_vec = _mm256_loadu_ps(&output_channel[i]);
                    __m256 result_vec = _mm256_add_ps(current_vec, bias_vec);
                    _mm256_storeu_ps(&output_channel[i], result_vec);
                } else {
                    // Handle remaining elements
                    for (size_t j = i; j < out_size; ++j) {
                        output_channel[j] += bias_val;
                    }
                    break;
                }
            }
        }
    }

    return output;
}

// SIMD-optimized bias gradient computation
static inline void compute_bias_grad_simd(float* __restrict bias_grad, const float* __restrict grad_output,
                                         int batch_size, int output_channels,
                                         int output_height, int output_width) {
    const int plane = output_height * output_width;
    const int batch_stride = output_channels * plane;

    #pragma omp parallel for schedule(static) if (output_channels * batch_size > 1)
    for (int oc = 0; oc < output_channels; ++oc) {
        __m256 vsum = _mm256_setzero_ps();
        float tail = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            const float* base = grad_output + b * batch_stride + oc * plane;
            int i = 0;
            int vec_end = (plane / 8) * 8;
            for (; i < vec_end; i += 8) {
                __m256 v = _mm256_loadu_ps(base + i);
                vsum = _mm256_add_ps(vsum, v);
            }
            for (; i < plane; ++i) tail += base[i];
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vsum);
        bias_grad[oc] = tail + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    }
}

// Generic weight gradient computation for any kernel size
static inline void compute_weight_grad_generic(float* __restrict weight_grad, const float* __restrict input, const float* __restrict grad_output,
                                             int batch_size, int input_channels, int output_channels,
                                             int input_height, int input_width,
                                             int output_height, int output_width,
                                             int kernel_size, int stride, int padding) {
    const int in_plane  = input_height * input_width;
    const int out_plane = output_height * output_width;
    const int in_batch_stride  = input_channels * in_plane;
    const int out_batch_stride = output_channels * out_plane;
    const int kernel_elements = kernel_size * kernel_size;

    #pragma omp parallel for collapse(3) schedule(static) if (output_channels * input_channels * kernel_elements > 1)
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int k_idx = 0; k_idx < kernel_elements; ++k_idx) {
                const int kh = k_idx / kernel_size;
                const int kw = k_idx % kernel_size;

                float sum = 0.0f;
                for (int b = 0; b < batch_size; ++b) {
                    const float* in_base = input + b * in_batch_stride + ic * in_plane;
                    const float* go_base = grad_output + b * out_batch_stride + oc * out_plane;

                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            const int ih = oh * stride - padding + kh;
                            const int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                sum += in_base[ih * input_width + iw] * go_base[oh * output_width + ow];
                            }
                        }
                    }
                }

                const int widx = oc * (input_channels * kernel_elements) + ic * kernel_elements + k_idx;
                weight_grad[widx] = sum;
            }
        }
    }
}

// Optimized weight gradient computation for 3x3 kernels
static inline void compute_weight_grad_3x3(float* __restrict weight_grad, const float* __restrict input, const float* __restrict grad_output,
                                         int batch_size, int input_channels, int output_channels,
                                         int input_height, int input_width,
                                         int output_height, int output_width) {
    const int in_plane  = input_height * input_width;
    const int out_plane = output_height * output_width;
    const int in_batch_stride  = input_channels * in_plane;
    const int out_batch_stride = output_channels * out_plane;

    #pragma omp parallel for collapse(2) schedule(static) if (output_channels * input_channels > 1)
    for (int oc = 0; oc < output_channels; ++oc) {
        for (int ic = 0; ic < input_channels; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
                int dh = kh - 1;
                int oh_start = (dh == -1) ? 1 : 0;
                int oh_end   = (dh ==  1) ? (output_height - 1) : output_height;
                for (int kw = 0; kw < 3; ++kw) {
                    int dw = kw - 1;
                    int ow_start = (dw == -1) ? 1 : 0;
                    int ow_end   = (dw ==  1) ? (output_width - 1) : output_width;

                    __m256 vsum = _mm256_setzero_ps();
                    float tail = 0.0f;

                    for (int b = 0; b < batch_size; ++b) {
                        const float* in_base = input + b * in_batch_stride + ic * in_plane;
                        const float* go_base = grad_output + b * out_batch_stride + oc * out_plane;
                        for (int oh = oh_start; oh < oh_end; ++oh) {
                            const int ih = oh + dh;
                            const int in_row_off = ih * input_width;
                            const int go_row_off = oh * output_width;
                            int ow = ow_start;
                            int vec_end = ow_start + ((ow_end - ow_start) / 8) * 8;
                            for (; ow < vec_end; ow += 8) {
                                __m256 vin = _mm256_loadu_ps(in_base + in_row_off + (ow + dw));
                                __m256 vgo = _mm256_loadu_ps(go_base + go_row_off + ow);
                                vsum = _mm256_fmadd_ps(vin, vgo, vsum);
                            }
                            for (; ow < ow_end; ++ow) {
                                tail += in_base[in_row_off + (ow + dw)] * go_base[go_row_off + ow];
                            }
                        }
                    }

                    float tmp[8];
                    _mm256_storeu_ps(tmp, vsum);
                    float sum = tail + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    const int widx = oc * (input_channels * 9) + ic * 9 + kh * 3 + kw;
                    weight_grad[widx] = sum;
                }
            }
        }
    }
}

// Optimized input gradient computation for 3x3 kernels (reverted to original efficient structure)
static inline void compute_input_grad_3x3(float* __restrict grad_input, const float* __restrict weights, const float* __restrict grad_output,
                                        int batch_size, int input_channels, int output_channels,
                                        int input_height, int input_width,
                                        int output_height, int output_width) {
    memset(grad_input, 0, (size_t)batch_size * input_channels * input_height * input_width * sizeof(float));

    const int in_plane  = input_height * input_width;
    const int out_plane = output_height * output_width;
    const int in_batch_stride  = input_channels * in_plane;
    const int out_batch_stride = output_channels * out_plane;

    // Revert to original loop order for better cache locality - this was the main issue
    #pragma omp parallel for collapse(2) schedule(static) if (batch_size * output_channels > 1)
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            const float* go_base = grad_output + b * out_batch_stride + oc * out_plane;
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    const float go = go_base[oh * output_width + ow];
                    const int ih_center = oh; // stride=1, padding=1
                    const int iw_center = ow;
                    for (int ic = 0; ic < input_channels; ++ic) {
                        const float* wbase = weights + oc * (input_channels * 9) + ic * 9;
                        float* gi_base = grad_input + b * in_batch_stride + ic * in_plane;

                        int ih0 = ih_center - 1, ih1 = ih_center, ih2 = ih_center + 1;
                        int iw0 = iw_center - 1, iw1 = iw_center, iw2 = iw_center + 1;

                        if (ih0 >= 0) {
                            if (iw0 >= 0) gi_base[ih0 * input_width + iw0] += wbase[0] * go;
                            gi_base[ih0 * input_width + iw1] += wbase[1] * go;
                            if (iw2 < input_width) gi_base[ih0 * input_width + iw2] += wbase[2] * go;
                        }
                        {
                            if (iw0 >= 0) gi_base[ih1 * input_width + iw0] += wbase[3] * go;
                            gi_base[ih1 * input_width + iw1] += wbase[4] * go;
                            if (iw2 < input_width) gi_base[ih1 * input_width + iw2] += wbase[5] * go;
                        }
                        if (ih2 < input_height) {
                            if (iw0 >= 0) gi_base[ih2 * input_width + iw0] += wbase[6] * go;
                            gi_base[ih2 * input_width + iw1] += wbase[7] * go;
                            if (iw2 < input_width) gi_base[ih2 * input_width + iw2] += wbase[8] * go;
                        }
                    }
                }
            }
        }
    }
}

Tensor* conv2D_backward(Conv2D* conv, Tensor* input, Tensor* grad_output) {
    if (!conv || !input || !grad_output) return NULL;

    memset(conv->weight_grad->data, 0, conv->weight_grad->size * sizeof(float));
    memset(conv->bias_grad->data, 0, conv->bias_grad->size * sizeof(float));

    const int batch_size = input->shape[0];
    const int input_channels = input->shape[1];
    const int input_height = input->shape[2];
    const int input_width = input->shape[3];
    const int output_channels = grad_output->shape[1];
    const int output_height = grad_output->shape[2];
    const int output_width = grad_output->shape[3];

    // Compute bias gradient (sum over batch, height, width for each output channel)
    compute_bias_grad_simd(conv->bias_grad->data, grad_output->data,
                          batch_size, output_channels, output_height, output_width);

    if (conv->kernel_size == 3 && conv->stride == 1 && conv->padding == 1) {
        compute_weight_grad_3x3(conv->weight_grad->data, input->data, grad_output->data,
                               batch_size, input_channels, output_channels,
                               input_height, input_width, output_height, output_width);
    } else {
        compute_weight_grad_generic(conv->weight_grad->data, input->data, grad_output->data,
                                   batch_size, input_channels, output_channels,
                                   input_height, input_width, output_height, output_width,
                                   conv->kernel_size, conv->stride, conv->padding);
    }

    int grad_input_shape[] = {batch_size, input_channels, input_height, input_width};
    Tensor* grad_input = tensor_create(grad_input_shape, 4);
    if (!grad_input) return NULL;

    if (conv->kernel_size == 3 && conv->stride == 1 && conv->padding == 1) {
        // Use optimized 3x3 version
        compute_input_grad_3x3(grad_input->data, conv->weight->data, grad_output->data,
                              batch_size, input_channels, output_channels,
                              input_height, input_width, output_height, output_width);
    } else {
        // Generic input gradient computation for any kernel size
        memset(grad_input->data, 0, (size_t)batch_size * input_channels * input_height * input_width * sizeof(float));

        const int in_plane  = input_height * input_width;
        const int out_plane = output_height * output_width;
        const int in_batch_stride  = input_channels * in_plane;
        const int out_batch_stride = output_channels * out_plane;

        #pragma omp parallel for collapse(2) schedule(static) if (batch_size * input_channels > 1)
        for (int b = 0; b < batch_size; ++b) {
            for (int ic = 0; ic < input_channels; ++ic) {
                float* gi_base = grad_input->data + b * in_batch_stride + ic * in_plane;

                for (int oc = 0; oc < output_channels; ++oc) {
                    const float* go_base = grad_output->data + b * out_batch_stride + oc * out_plane;
                    const float* wbase = conv->weight->data + oc * (input_channels * conv->kernel_size * conv->kernel_size) + ic * conv->kernel_size * conv->kernel_size;

                    for (int oh = 0; oh < output_height; ++oh) {
                        for (int ow = 0; ow < output_width; ++ow) {
                            const float go = go_base[oh * output_width + ow];
                            const int ih_center = oh * conv->stride - conv->padding;
                            const int iw_center = ow * conv->stride - conv->padding;

                            for (int kh = 0; kh < conv->kernel_size; ++kh) {
                                for (int kw = 0; kw < conv->kernel_size; ++kw) {
                                    const int ih = ih_center + kh;
                                    const int iw = iw_center + kw;

                                    if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                        const float w = wbase[kh * conv->kernel_size + kw];
                                        gi_base[ih * input_width + iw] += w * go;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    return grad_input;
}

void conv2D_zero_grad(Conv2D* conv) {
    if (conv && conv->weight_grad && conv->bias_grad) {
        memset(conv->weight_grad->data, 0, conv->weight_grad->size * sizeof(float));
        memset(conv->bias_grad->data, 0, conv->bias_grad->size * sizeof(float));
    }
}
