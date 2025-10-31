#include "nn/layers/linear.h"
#include "nn/core/layer_grad.h"
#include "nn/core/init.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <cblas.h>

#define M_PI 3.14159265358979323846

Linear* linear_create(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) {
        printf("Error: Input and output sizes must be positive\n");
        return NULL;
    }

    Linear* layer = (Linear*)malloc(sizeof(Linear));
    if (!layer) return NULL;

    layer->input_size = input_size;
    layer->output_size = output_size;

    int weights_shape[2] = {output_size, input_size};
    int biases_shape[2] = {output_size, 1};
    Tensor *weights = tensor_create(weights_shape, 2);
    Tensor *biases = tensor_create(biases_shape, 2);
    layer->layer_grad = layer_grad_create(weights, biases);

    init_kaiming_normal(weights);
    memset(biases->data, 0, biases->size * sizeof(float));
    layer->input_cache = NULL;

    return layer;
}

void linear_free(Linear* layer) {
    if (layer) {
        if (layer->layer_grad) layer_grad_free(layer->layer_grad);
        if (layer->input_cache) tensor_free(layer->input_cache);
        free(layer);
    }
}

LinearOutput* linear_forward(Linear* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    if (input->ndim != 2) {
        printf("Error: Linear layer expects 2D input (batch_size, input_size), got %dD\n", input->ndim);
        return NULL;
    }

    int batch_size = input->shape[0];
    int input_size = input->shape[1];

    if (input_size != layer->input_size) {
        printf("Error: Input size %d doesn't match layer input size %d\n",
               input_size, layer->input_size);
        return NULL;
    }

    int output_shape[2] = {batch_size, layer->output_size};
    Tensor* output = tensor_create(output_shape, 2);
    if (!output) return NULL;

    if (layer->input_cache) tensor_free(layer->input_cache);
    int input_cache_shape[2] = {batch_size, input_size};
    layer->input_cache = tensor_create(input_cache_shape, 2);
    if (!layer->input_cache) {
        tensor_free(output);
        return NULL;
    }

    // Copy input to cache
    memcpy(layer->input_cache->data, input->data, input->size * sizeof(float));

    // Perform matrix multiplication: output = input @ weights + bias
    // Note: This assumes weights are stored in row-major order
    // For proper linear layer behavior, we might need weights.T depending on convention
    Tensor* temp_output = tensor_matmul(layer->input_cache, layer->layer_grad->weights);
    if (!temp_output) {
        tensor_free(output);
        return NULL;
    }

    // Copy result to output
    memcpy(output->data, temp_output->data, output->size * sizeof(float));
    tensor_free(temp_output);

    // Add bias to each row of output using tensor operation
    tensor_add_bias_inplace(output, layer->layer_grad->biases);

    // Create result structure
    LinearOutput* result = (LinearOutput*)malloc(sizeof(LinearOutput));
    if (!result) {
        tensor_free(output);
        return NULL;
    }

    result->output = output;
    result->layer = layer;

    return result;
}

// Free result structure
void linear_output_free(LinearOutput* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        // Don't free layer here, it's managed separately
        free(result);
    }
}

// Backward pass implementation
LinearBackwardOutput* linear_backward(Linear* layer, LinearOutput* forward_result,
                                     Tensor* output_grad) {
    if (!layer || !forward_result || !output_grad || !layer->input_cache) return NULL;

    // Check output gradient dimensions - expect 2D tensor (batch_size, output_size)
    if (output_grad->ndim != 2) {
        printf("Error: Output gradient should be 2D (batch_size, output_size), got %dD\n", output_grad->ndim);
        return NULL;
    }

    int batch_size = output_grad->shape[0];
    int output_size = output_grad->shape[1];

    if (batch_size != forward_result->output->shape[0] || output_size != layer->output_size) {
        printf("Error: Output gradient dimensions don't match\n");
        return NULL;
    }

    // Create input gradient tensor (batch_size, input_size)
    int input_grad_shape[2] = {batch_size, layer->input_size};
    Tensor* input_grad = tensor_create(input_grad_shape, 2);
    if (!input_grad) return NULL;

    // Input gradients are zeroed in input_grad_simd function

    // Zero out weight gradients
    if (layer->layer_grad && layer->layer_grad->weight_grad) {
        memset(layer->layer_grad->weight_grad->data, 0, layer->layer_grad->weight_grad->size * sizeof(float));
    }

    // Compute gradients using tensor operations
    // dL/dW = dL/dy.T @ input (gives output_size x input_size)
    // dL/db = sum(dL/dy, axis=0)
    // dL/dx = dL/dy @ W.T

    // Weight gradients: output_grad.T @ input
    tensor_outer_product_accumulate(layer->layer_grad->weight_grad, output_grad, layer->input_cache);

    // Bias gradients: sum over batch dimension (axis 0)
    Tensor* bias_grad_tensor = tensor_sum_axis(output_grad, 0);
    if (bias_grad_tensor) {
        memcpy(layer->layer_grad->bias_grad->data, bias_grad_tensor->data,
               layer->layer_grad->bias_grad->size * sizeof(float));
        tensor_free(bias_grad_tensor);
    }

    // Input gradients: output_grad @ weights.T
    // Alternative using BLAS: input_grad = output_grad @ weights.T
    // input_grad_simd(input_grad->data, output_grad->data, layer->layer_grad->weights->data,
    //                batch_size, layer->input_size, layer->output_size);

    // Using tensor operations with transpose
    Tensor* weights_T = tensor_transpose(layer->layer_grad->weights);
    if (weights_T) {
        Tensor* input_grad_tensor = tensor_matmul(output_grad, weights_T);
        if (input_grad_tensor) {
            memcpy(input_grad->data, input_grad_tensor->data, input_grad->size * sizeof(float));
            tensor_free(input_grad_tensor);
        }
        tensor_free(weights_T);
    }

    // Create result structure
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

