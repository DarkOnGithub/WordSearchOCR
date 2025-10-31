#include "dropout2d.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Static function for random number generation
static float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

// Constructor
Dropout2D* dropout2d_create(float dropout_rate) {
    if (dropout_rate < 0.0f || dropout_rate >= 1.0f) {
        printf("Error: Dropout2D rate must be in range [0.0, 1.0)\n");
        return NULL;
    }

    Dropout2D* layer = (Dropout2D*)malloc(sizeof(Dropout2D));
    if (!layer) return NULL;

    layer->dropout_rate = dropout_rate;
    layer->training = true;  // Default to training mode

    // Seed random number generator
    srand((unsigned int)time(NULL));

    return layer;
}

// Destructor
void dropout2d_free(Dropout2D* layer) {
    if (layer) {
        free(layer);
    }
}

// Mode switching
void dropout2d_train(Dropout2D* layer) {
    if (layer) {
        layer->training = true;
    }
}

void dropout2d_eval(Dropout2D* layer) {
    if (layer) {
        layer->training = false;
    }
}

// Forward pass
Dropout2DResult* dropout2d_forward(Dropout2D* layer, Tensor* input) {
    if (!layer || !input) return NULL;

    // Create output tensor with same dimensions as input
    Tensor* output = tensor_create(input->batch_size, input->channels,
                                  input->height, input->width);
    if (!output) return NULL;

    Tensor* mask = NULL;

    if (layer->training && layer->dropout_rate > 0.0f) {
        // Training mode: apply dropout2d
        // Create mask with shape (batch_size, channels, 1, 1)
        mask = tensor_create(input->batch_size, input->channels, 1, 1);
        if (!mask) {
            tensor_free(output);
            return NULL;
        }

        // Generate dropout mask for each channel in each batch
        for (int b = 0; b < input->batch_size; ++b) {
            for (int c = 0; c < input->channels; ++c) {
                float rand_val = random_float();
                int mask_idx = b * input->channels + c;

                if (rand_val < layer->dropout_rate) {
                    // Drop this entire channel
                    mask->data[mask_idx] = 0.0f;
                } else {
                    // Keep this channel, scale by (1/(1-dropout_rate)) for variance preservation
                    mask->data[mask_idx] = 1.0f / (1.0f - layer->dropout_rate);
                }
            }
        }

        // Apply mask to output tensor by broadcasting
        for (int b = 0; b < input->batch_size; ++b) {
            for (int c = 0; c < input->channels; ++c) {
                int mask_idx = b * input->channels + c;
                float mask_val = mask->data[mask_idx];

                // Apply mask to all spatial positions in this channel
                for (int h = 0; h < input->height; ++h) {
                    for (int w = 0; w < input->width; ++w) {
                        int input_idx = ((b * input->channels + c) * input->height + h) * input->width + w;
                        output->data[input_idx] = input->data[input_idx] * mask_val;
                    }
                }
            }
        }
    } else {
        // Inference mode or dropout_rate = 0: scale by (1-dropout_rate)
        float scale = 1.0f - layer->dropout_rate;
        for (int i = 0; i < input->size; ++i) {
            output->data[i] = input->data[i] * scale;
        }
    }

    // Create result structure
    Dropout2DResult* result = (Dropout2DResult*)malloc(sizeof(Dropout2DResult));
    if (!result) {
        tensor_free(output);
        if (mask) tensor_free(mask);
        return NULL;
    }

    result->output = output;
    result->mask = mask;

    return result;
}

// Free result structure
void dropout2d_result_free(Dropout2DResult* result) {
    if (result) {
        if (result->output) tensor_free(result->output);
        if (result->mask) tensor_free(result->mask);
        free(result);
    }
}

// Backward pass
Dropout2DBackwardResult* dropout2d_backward(Dropout2D* layer, Dropout2DResult* forward_result,
                                           Tensor* output_grad) {
    if (!layer || !forward_result || !forward_result->output || !output_grad) return NULL;

    // Check if output_grad dimensions match forward result output dimensions
    if (output_grad->batch_size != forward_result->output->batch_size ||
        output_grad->channels != forward_result->output->channels ||
        output_grad->height != forward_result->output->height ||
        output_grad->width != forward_result->output->width) {
        printf("Error: Output gradient dimensions don't match forward output dimensions got %d x %d x %d x %d expected %d x %d x %d x %d\n", output_grad->batch_size, output_grad->channels, output_grad->height, output_grad->width, forward_result->output->batch_size, forward_result->output->channels, forward_result->output->height, forward_result->output->width);
        return NULL;
    }

    // Create input gradient tensor with same dimensions as forward input
    Tensor* input_grad = tensor_create(output_grad->batch_size, output_grad->channels,
                                       output_grad->height, output_grad->width);
    if (!input_grad) return NULL;

    if (layer->training && forward_result->mask) {
        // Training mode: use the dropout mask to route gradients
        // Broadcast mask from (batch_size, channels, 1, 1) to full tensor shape
        for (int b = 0; b < output_grad->batch_size; ++b) {
            for (int c = 0; c < output_grad->channels; ++c) {
                int mask_idx = b * output_grad->channels + c;
                float mask_val = forward_result->mask->data[mask_idx];

                // Apply mask to all spatial positions in this channel
                for (int h = 0; h < output_grad->height; ++h) {
                    for (int w = 0; w < output_grad->width; ++w) {
                        int grad_idx = ((b * output_grad->channels + c) * output_grad->height + h) * output_grad->width + w;
                        input_grad->data[grad_idx] = output_grad->data[grad_idx] * mask_val;
                    }
                }
            }
        }
    } else {
        // Inference mode or no mask: scale gradients by (1-dropout_rate)
        float scale = 1.0f - layer->dropout_rate;
        for (int i = 0; i < output_grad->size; ++i) {
            input_grad->data[i] = output_grad->data[i] * scale;
        }
    }

    // Create result structure
    Dropout2DBackwardResult* result = (Dropout2DBackwardResult*)malloc(sizeof(Dropout2DBackwardResult));
    if (!result) {
        tensor_free(input_grad);
        return NULL;
    }

    result->input_grad = input_grad;
    return result;
}

// Free backward result structure
void dropout2d_backward_result_free(Dropout2DBackwardResult* result) {
    if (result) {
        if (result->input_grad) tensor_free(result->input_grad);
        free(result);
    }
}

// Utility functions
float dropout2d_get_rate(Dropout2D* layer) {
    return layer ? layer->dropout_rate : 0.0f;
}

bool dropout2d_is_training(Dropout2D* layer) {
    return layer ? layer->training : false;
}
