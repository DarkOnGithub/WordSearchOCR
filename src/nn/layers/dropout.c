#include "nn/layers/dropout.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "nn/core/utils.h"

Dropout* dropout_create(float dropout_rate) {
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        printf("Error: Dropout rate must be in range [0.0, 1.0)\n");
        return NULL;
    }

    Dropout* layer = (Dropout*)malloc(sizeof(Dropout));
    if (!layer) return NULL;

    layer->dropout_rate = dropout_rate;
    layer->training = true;

    return layer;
}

void dropout_free(Dropout* layer) {
    if (layer) {
        free(layer);
    }
}

DropoutOutput* dropout_forward(Dropout* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    Tensor* output = tensor_create(input->shape, input->ndim);
    if (!output) return NULL;

    Tensor* mask = NULL;

    if (layer->training && layer->dropout_rate > 0.0f) {
        mask = tensor_create(input->shape, input->ndim);
        if (!mask) {
            tensor_free(output);
            return NULL;
        }

        for (int i = 0; i < input->size; ++i) {
            float rand_val = random_float();
            if (rand_val < layer->dropout_rate) {
                mask->data[i] = 0.0f;
                output->data[i] = 0.0f;
            } else {
                mask->data[i] = 1.0f / (1.0f - layer->dropout_rate);
                output->data[i] = input->data[i] * mask->data[i];
            }
        }
    } else {
        float scale = 1.0f - layer->dropout_rate;
        for (int i = 0; i < input->size; ++i) {
            output->data[i] = input->data[i] * scale;
        }
    }

    DropoutOutput* result = (DropoutOutput*)malloc(sizeof(DropoutOutput));
    if (!result) {
        tensor_free(output);
        if (mask) tensor_free(mask);
        return NULL;
    }

    result->output = output;
    result->mask = mask;

    return result;
}

void dropout_output_free(DropoutOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        if (result->mask) tensor_free(result->mask);
        free(result);
    }
}

DropoutBackwardOutput* dropout_backward(Dropout* layer, DropoutOutput* forward_result,
                                       Tensor* output_grad) {
    if (!layer || !forward_result || !forward_result->output || !output_grad) return NULL;

    if (output_grad->shape[0] != forward_result->output->shape[0] ||
        output_grad->shape[1] != forward_result->output->shape[1]) {
        printf("Error: Output gradient dimensions don't match forward output dimensions\n");
        return NULL;
    }

    Tensor* input_grad = tensor_create(output_grad->shape, output_grad->ndim);
    if (!input_grad) return NULL;

    if (layer->training && forward_result->mask) {
        for (int i = 0; i < output_grad->size; i++) {
            input_grad->data[i] = output_grad->data[i] * forward_result->mask->data[i];
        }
    } else {
        for (int i = 0; i < output_grad->size; i++) {
            input_grad->data[i] = output_grad->data[i];
        }
    }

    DropoutBackwardOutput* result = (DropoutBackwardOutput*)malloc(sizeof(DropoutBackwardOutput));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

void dropout_backward_output_free(DropoutBackwardOutput* result) {
    if (result) {
        tensor_free(result->input_grad);
        free(result);
    }
}
