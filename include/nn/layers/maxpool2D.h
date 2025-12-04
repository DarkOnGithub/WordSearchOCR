#pragma once
#include <stdbool.h>
#include "nn/core/tensor.h"

typedef struct {
    int kernel_size_h;
    int kernel_size_w;
    int stride_h;
    int stride_w;
    int padding_h;
    int padding_w;
    int dilation_h;
    int dilation_w;
    bool return_indices;
    bool ceil_mode;
} MaxPool2D;

typedef struct {
    Tensor* output;
    Tensor* indices;
    Tensor* input;
} MaxPool2DOutput;

typedef struct {
    Tensor* input_grad;
} MaxPool2DBackwardOutput;

MaxPool2D* maxpool2d_create(int kernel_size_h, int kernel_size_w,
                          int stride_h, int stride_w,
                          int padding_h, int padding_w,
                          int dilation_h, int dilation_w,
                          bool return_indices, bool ceil_mode);
void maxpool2d_free(MaxPool2D* layer);

MaxPool2D* maxpool2d_create_square(int kernel_size, int stride, int padding,
                                 bool return_indices, bool ceil_mode);
MaxPool2D* maxpool2d_create_simple(int kernel_size, int stride);

MaxPool2DOutput* maxpool2d_forward(MaxPool2D* layer, Tensor* input);
void maxpool2d_output_free(MaxPool2DOutput* result);

MaxPool2DBackwardOutput* maxpool2d_backward(MaxPool2D* layer, MaxPool2DOutput* forward_result,
                                          Tensor* output_grad);
void maxpool2d_backward_output_free(MaxPool2DBackwardOutput* result);
