#pragma once
#include "nn/core/tensor.h"

// Cross Entropy Loss structure (stateless, no learnable parameters)
typedef struct {

} CrossEntropyLoss;

typedef struct {
    float loss;
    Tensor* softmax_output;
    Tensor* targets;
} CrossEntropyOutput;

typedef struct {
    Tensor* input_grad;
} CrossEntropyBackwardOutput;

CrossEntropyLoss* cross_entropy_loss_create();
void cross_entropy_loss_free(CrossEntropyLoss* loss);
CrossEntropyOutput* cross_entropy_loss_forward(CrossEntropyLoss* loss, Tensor* input, Tensor* targets);
void cross_entropy_result_free(CrossEntropyOutput* result);
CrossEntropyBackwardOutput* cross_entropy_loss_backward(CrossEntropyLoss* loss,
                                                        CrossEntropyOutput* forward_result,
                                                        Tensor* output_grad);
void cross_entropy_backward_result_free(CrossEntropyBackwardOutput* result);
Tensor* softmax(Tensor* input);
