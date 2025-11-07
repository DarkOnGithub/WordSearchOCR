#define _POSIX_C_SOURCE 199309L
#include "../include/nn/cnn.h"
#include "../include/nn/nn/silu.h"
#include "../include/nn/layers/batch_norm2d.h"
#include "../include/nn/layers/adaptive_avg_pool2D.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct timespec TimingType;
#define GET_TIME(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define TIME_DIFF(start, end) ((double)((end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec)) / 1000000.0) // ms
#define INIT_TIMING()

static int timing_initialized = 0;

//Useful to detect gradient explosion / vanishing
//!WARNING: DO NOT USE IN PROD
void check_gradients(const char* label, Tensor* grad) {
    return;
    if (!grad) {
        return;
    }

    float max_grad = 0.0f;
    float min_grad = INFINITY;
    float avg_grad = 0.0f;
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;

    for (int i = 0; i < grad->size; i++) {
        float g = grad->data[i];
        if (isnan(g)) nan_count++;
        else if (isinf(g)) inf_count++;
        else {
            float abs_g = fabsf(g);
            if (abs_g > max_grad) max_grad = abs_g;
            if (abs_g < min_grad && abs_g > 0) min_grad = abs_g;
            if (abs_g == 0.0f) zero_count++;
            avg_grad += abs_g;
        }
    }
    printf("DEBUG: %s - Max grad: %f, Min grad: %f, Avg grad: %f, Nan count: %d, Inf count: %d, Zero count: %d\n", label, max_grad, min_grad, avg_grad, nan_count, inf_count, zero_count);

}

CNN* cnn_create() {
    CNN* model = (CNN*)malloc(sizeof(CNN));
    if (!model) {
        fprintf(stderr, "Failed to allocate CNN model\n");
        return NULL;
    }

    // Block 1: 3x3 -> 1x1 with residual
    model->conv1_3x3 = conv2D_create(1, 32, 3, 1, 1);      // 1->32, 3x3, stride=1, padding=1
    model->bn1_3x3 = batch_norm2d_create(32, 0.1f, 1e-5f); // Batch norm for 32 channels
    model->conv1_1x1 = conv2D_create(32, 64, 1, 1, 0);     // 32->64, 1x1, stride=1, padding=0
    model->bn1_1x1 = batch_norm2d_create(64, 0.1f, 1e-5f); // Batch norm for 64 channels
    model->shortcut1 = conv2D_create(1, 64, 1, 1, 0);      // 1->64, 1x1 shortcut

    // Block 2: 3x3 -> 1x1 with residual
    model->conv2_3x3 = conv2D_create(64, 64, 3, 1, 1);     // 64->64, 3x3, stride=1, padding=1
    model->bn2_3x3 = batch_norm2d_create(64, 0.1f, 1e-5f); // Batch norm for 64 channels
    model->conv2_1x1 = conv2D_create(64, 128, 1, 1, 0);    // 64->128, 1x1, stride=1, padding=0
    model->bn2_1x1 = batch_norm2d_create(128, 0.1f, 1e-5f);// Batch norm for 128 channels
    model->shortcut2 = conv2D_create(64, 128, 1, 1, 0);    // 64->128, 1x1 shortcut

    // Block 3: 3x3 -> 1x1 with residual
    model->conv3_3x3 = conv2D_create(128, 128, 3, 1, 1);   // 128->128, 3x3, stride=1, padding=1
    model->bn3_3x3 = batch_norm2d_create(128, 0.1f, 1e-5f);// Batch norm for 128 channels
    model->conv3_1x1 = conv2D_create(128, 256, 1, 1, 0);   // 128->256, 1x1, stride=1, padding=0
    model->bn3_1x1 = batch_norm2d_create(256, 0.1f, 1e-5f);// Batch norm for 256 channels
    model->shortcut3 = conv2D_create(128, 256, 1, 1, 0);   // 128->256, 1x1 shortcut

    // Pooling layers (only for blocks 1 and 2)
    model->pool1 = maxpool2d_create_simple(2, 2);         // 28x28 -> 14x14
    model->pool2 = maxpool2d_create_simple(2, 2);         // 14x14 -> 7x7
    model->gap = adaptive_avg_pool2d_create(1, 1);       // 7x7 -> 1x1

    // Dropout layers (adjusted rates to match PyTorch)
    model->dropout_conv = dropout2d_create(0.10f);       // 0.10 for conv layers
    model->dropout_fc = dropout_create(0.25f);           // 0.25 for fc layers

    // Fully connected layers (adjusted input size: 256*1*1 = 256)
    model->fc1 = linear_create(256, 128);                // 256 -> 128
    model->fc2 = linear_create(128, 26);                 // 128 -> 26

    model->optimizer = NULL;
    model->scheduler = NULL;
    model->criterion = cross_entropy_loss_create();

    model->training = true;

    model->timing_verbose = false;

    return model;
}

void cnn_free(CNN* model) {
    if (!model) return;

    // Free convolutional layers and batch norms - Block 1
    conv2D_free(model->conv1_3x3);
    batch_norm2d_free(model->bn1_3x3);
    conv2D_free(model->conv1_1x1);
    batch_norm2d_free(model->bn1_1x1);
    conv2D_free(model->shortcut1);

    // Free convolutional layers and batch norms - Block 2
    conv2D_free(model->conv2_3x3);
    batch_norm2d_free(model->bn2_3x3);
    conv2D_free(model->conv2_1x1);
    batch_norm2d_free(model->bn2_1x1);
    conv2D_free(model->shortcut2);

    // Free convolutional layers and batch norms - Block 3
    conv2D_free(model->conv3_3x3);
    batch_norm2d_free(model->bn3_3x3);
    conv2D_free(model->conv3_1x1);
    batch_norm2d_free(model->bn3_1x1);
    conv2D_free(model->shortcut3);

    // Free pooling layers
    maxpool2d_free(model->pool1);
    maxpool2d_free(model->pool2);
    adaptive_avg_pool2d_free(model->gap);

    // Free dropout layers
    dropout2d_free(model->dropout_conv);
    dropout_free(model->dropout_fc);

    // Free fully connected layers
    linear_free(model->fc1);
    linear_free(model->fc2);

    // Free training components
    if (model->optimizer) adam_free(model->optimizer);
    if (model->scheduler) step_lr_free(model->scheduler);
    cross_entropy_loss_free(model->criterion);

    free(model);
}

void cnn_train(CNN* model) {
    model->training = true;
    // Set batch norm layers to training mode
    batch_norm2d_set_training(model->bn1_3x3, true);
    batch_norm2d_set_training(model->bn1_1x1, true);
    batch_norm2d_set_training(model->bn2_3x3, true);
    batch_norm2d_set_training(model->bn2_1x1, true);
    batch_norm2d_set_training(model->bn3_3x3, true);
    batch_norm2d_set_training(model->bn3_1x1, true);
    model->dropout_conv->training = true;
    model->dropout_fc->training = true;
}

void cnn_eval(CNN* model) {
    model->training = false;
    // Set batch norm layers to eval mode
    batch_norm2d_set_training(model->bn1_3x3, false);
    batch_norm2d_set_training(model->bn1_1x1, false);
    batch_norm2d_set_training(model->bn2_3x3, false);
    batch_norm2d_set_training(model->bn2_1x1, false);
    batch_norm2d_set_training(model->bn3_3x3, false);
    batch_norm2d_set_training(model->bn3_1x1, false);
    model->dropout_conv->training = false;
    model->dropout_fc->training = false;
}

CNNForwardResult* cnn_forward(CNN* model, Tensor* input) {
    if (!timing_initialized) {
        INIT_TIMING();
        timing_initialized = 1;
    }

    TimingType start_total, end_total, start_op, end_op;
    GET_TIME(start_total);

    CNNForwardResult* result = (CNNForwardResult*)malloc(sizeof(CNNForwardResult));
    if (!result) {
        fprintf(stderr, "Failed to allocate forward result\n");
        return NULL;
    }
    result->input = input;

    // --- Block 1: 28x28 -> 14x14 (with residual connection) ---
    // Shortcut path
    GET_TIME(start_op);
    result->shortcut1_out = conv2D_forward(model->shortcut1, input);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Shortcut1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path: Conv3x3 -> BN -> SiLU -> Conv1x1 -> BN
    GET_TIME(start_op);
    result->conv1_3x3_out = conv2D_forward(model->conv1_3x3, input);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv1_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn1_3x3_result = batch_norm2d_forward(model->bn1_3x3, result->conv1_3x3_out);
    result->silu1_3x3 = silu(result->bn1_3x3_result->output);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN1_3x3 + SiLU1_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->conv1_1x1_out = conv2D_forward(model->conv1_1x1, result->silu1_3x3);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv1_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn1_1x1_result = batch_norm2d_forward(model->bn1_1x1, result->conv1_1x1_out);
    // No activation here - activation applied after residual addition
    result->residual1_out = result->bn1_1x1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN1_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual addition + activation
    GET_TIME(start_op);
    Tensor* residual1_sum = tensor_add(result->residual1_out, result->shortcut1_out);
    result->silu1_residual = silu(residual1_sum);
    tensor_free(residual1_sum);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Residual1 + SiLU: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Pooling and dropout
    GET_TIME(start_op);
    result->pool1_result = maxpool2d_forward(model->pool1, result->silu1_residual);
    result->pool1_out = result->pool1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Pool1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->dropout1_result = dropout2d_forward(model->dropout_conv, result->pool1_out);
    result->dropout1_out = result->dropout1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // --- Block 2: 14x14 -> 7x7 (with residual connection) ---
    // Shortcut path
    GET_TIME(start_op);
    result->shortcut2_out = conv2D_forward(model->shortcut2, result->dropout1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Shortcut2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path: Conv3x3 -> BN -> SiLU -> Conv1x1 -> BN
    GET_TIME(start_op);
    result->conv2_3x3_out = conv2D_forward(model->conv2_3x3, result->dropout1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv2_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn2_3x3_result = batch_norm2d_forward(model->bn2_3x3, result->conv2_3x3_out);
    result->silu2_3x3 = silu(result->bn2_3x3_result->output);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN2_3x3 + SiLU2_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->conv2_1x1_out = conv2D_forward(model->conv2_1x1, result->silu2_3x3);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv2_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn2_1x1_result = batch_norm2d_forward(model->bn2_1x1, result->conv2_1x1_out);
    result->residual2_out = result->bn2_1x1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN2_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual addition + activation
    GET_TIME(start_op);
    Tensor* residual2_sum = tensor_add(result->residual2_out, result->shortcut2_out);
    result->silu2_residual = silu(residual2_sum);
    tensor_free(residual2_sum);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Residual2 + SiLU: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Pooling and dropout
    GET_TIME(start_op);
    result->pool2_result = maxpool2d_forward(model->pool2, result->silu2_residual);
    result->pool2_out = result->pool2_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Pool2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->dropout2_result = dropout2d_forward(model->dropout_conv, result->pool2_out);
    result->dropout2_out = result->dropout2_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // --- Block 3: 7x7 -> 1x1 (with residual connection) ---
    // Shortcut path
    GET_TIME(start_op);
    result->shortcut3_out = conv2D_forward(model->shortcut3, result->dropout2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Shortcut3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path: Conv3x3 -> BN -> SiLU -> Conv1x1 -> BN
    GET_TIME(start_op);
    result->conv3_3x3_out = conv2D_forward(model->conv3_3x3, result->dropout2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv3_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn3_3x3_result = batch_norm2d_forward(model->bn3_3x3, result->conv3_3x3_out);
    result->silu3_3x3 = silu(result->bn3_3x3_result->output);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN3_3x3 + SiLU3_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->conv3_1x1_out = conv2D_forward(model->conv3_1x1, result->silu3_3x3);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv3_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->bn3_1x1_result = batch_norm2d_forward(model->bn3_1x1, result->conv3_1x1_out);
    result->residual3_out = result->bn3_1x1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - BN3_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual addition + activation
    GET_TIME(start_op);
    Tensor* residual3_sum = tensor_add(result->residual3_out, result->shortcut3_out);
    result->silu3_residual = silu(residual3_sum);
    tensor_free(residual3_sum);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Residual3 + SiLU: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Global average pooling instead of max pooling
    GET_TIME(start_op);
    result->gap_result = adaptive_avg_pool2d_forward(model->gap, result->silu3_residual);
    result->gap_out = result->gap_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - GAP: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // --- Flatten & FC ---
    GET_TIME(start_op);
    int flattened_shape[] = {result->gap_out->shape[0],
                             result->gap_out->shape[1] * result->gap_out->shape[2] * result->gap_out->shape[3]};
    Tensor* flattened_copy = tensor_create(flattened_shape, 2);
    memcpy(flattened_copy->data, result->gap_out->data, result->gap_out->size * sizeof(float));
    result->flattened = flattened_copy;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Flatten: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // FC layers
    GET_TIME(start_op);
    result->fc1_result = linear_forward(model->fc1, result->flattened);
    result->fc1_out = result->fc1_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - FC1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->silu1_out = silu(result->fc1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - SiLU_FC1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->dropout_fc_result = dropout_forward(model->dropout_fc, result->silu1_out);
    result->dropout_fc_out = result->dropout_fc_result->output;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout_FC: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    result->fc2_result = linear_forward(model->fc2, result->dropout_fc_out);
    result->fc2_out = result->fc2_result->output;
    check_gradients("Forward - FC2 output", result->fc2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - FC2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Forward - Total: %.3f ms\n", TIME_DIFF(start_total, end_total));
    }

    return result;
}

void cnn_forward_result_free(CNNForwardResult* result) {
    if (!result) return;

    // Free Block 1 outputs
    tensor_free(result->conv1_3x3_out);
    result->conv1_3x3_out = NULL;
    if (result->bn1_3x3_result) {
        batch_norm2d_output_free(result->bn1_3x3_result);
        result->bn1_3x3_result = NULL;
    }
    tensor_free(result->silu1_3x3);
    result->silu1_3x3 = NULL;
    tensor_free(result->conv1_1x1_out);
    result->conv1_1x1_out = NULL;
    if (result->bn1_1x1_result) {
        batch_norm2d_output_free(result->bn1_1x1_result);
        result->bn1_1x1_result = NULL;
        result->residual1_out = NULL;  // This was freed by the batch norm free
    }
    tensor_free(result->shortcut1_out);
    result->shortcut1_out = NULL;
    tensor_free(result->silu1_residual);
    result->silu1_residual = NULL;
    maxpool2d_output_free(result->pool1_result);
    dropout2d_output_free(result->dropout1_result);

    // Free Block 2 outputs
    tensor_free(result->conv2_3x3_out);
    result->conv2_3x3_out = NULL;
    if (result->bn2_3x3_result) {
        batch_norm2d_output_free(result->bn2_3x3_result);
        result->bn2_3x3_result = NULL;
    }
    tensor_free(result->silu2_3x3);
    result->silu2_3x3 = NULL;
    tensor_free(result->conv2_1x1_out);
    result->conv2_1x1_out = NULL;
    if (result->bn2_1x1_result) {
        batch_norm2d_output_free(result->bn2_1x1_result);
        result->bn2_1x1_result = NULL;
        result->residual2_out = NULL;  // This was freed by the batch norm free
    }
    tensor_free(result->shortcut2_out);
    result->shortcut2_out = NULL;
    tensor_free(result->silu2_residual);
    result->silu2_residual = NULL;
    maxpool2d_output_free(result->pool2_result);
    dropout2d_output_free(result->dropout2_result);

    // Free Block 3 outputs
    tensor_free(result->conv3_3x3_out);
    result->conv3_3x3_out = NULL;
    if (result->bn3_3x3_result) {
        batch_norm2d_output_free(result->bn3_3x3_result);
        result->bn3_3x3_result = NULL;
    }
     tensor_free(result->silu3_3x3);
     result->silu3_3x3 = NULL;
    tensor_free(result->conv3_1x1_out);
    result->conv3_1x1_out = NULL;
    if (result->bn3_1x1_result) {
        batch_norm2d_output_free(result->bn3_1x1_result);
        result->bn3_1x1_result = NULL;
        result->residual3_out = NULL;  // This was freed by the batch norm free
    }
    tensor_free(result->shortcut3_out);
    result->shortcut3_out = NULL;
    tensor_free(result->silu3_residual);
    result->silu3_residual = NULL;
    if (result->gap_result) {
        adaptive_avg_pool2d_output_free(result->gap_result);
    }

    tensor_free(result->flattened);
    result->flattened = NULL;
    if (result->fc1_result) {
        linear_output_free(result->fc1_result);
        result->fc1_result = NULL;
    }
    // Note: fc1_out is freed by linear_output_free above
    result->fc1_out = NULL;
    tensor_free(result->silu1_out);
    result->silu1_out = NULL;
    if (result->dropout_fc_result) {
        dropout_output_free(result->dropout_fc_result);
        result->dropout_fc_result = NULL;
    }
    if (result->fc2_result) {
        linear_output_free(result->fc2_result);
        result->fc2_result = NULL;
    }

    free(result);
}

Tensor* cnn_backward(CNN* model, CNNForwardResult* forward_result, CrossEntropyOutput* loss_result) {
    TimingType start_total, end_total, start_op, end_op;
    GET_TIME(start_total);

    // Get gradient w.r.t. logits using existing loss result
    // The loss is averaged over batch, so upstream gradient should be 1.0 for each sample
    GET_TIME(start_op);
    int loss_grad_shape[] = {forward_result->fc2_out->shape[0], forward_result->fc2_out->shape[1]};
    Tensor* loss_upstream_grad = tensor_create_ones(loss_grad_shape, 2);
    CrossEntropyBackwardOutput* loss_grad_result = cross_entropy_loss_backward(model->criterion, loss_result, loss_upstream_grad);
    Tensor* output_grad = loss_grad_result->input_grad;
    loss_grad_result->input_grad = NULL;  // Prevent double-free
    cross_entropy_backward_result_free(loss_grad_result);
    tensor_free(loss_upstream_grad);
    check_gradients("Loss backward", output_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Loss: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // FC2 backward
    GET_TIME(start_op);
    LinearBackwardOutput* fc2_back = linear_backward(model->fc2, forward_result->fc2_result, output_grad);
    Tensor* fc2_input_grad = fc2_back->input_grad;
    fc2_back->input_grad = NULL;
    linear_backward_output_free(fc2_back);
    check_gradients("FC2 backward", fc2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - FC2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Dropout FC backward
    GET_TIME(start_op);
    DropoutBackwardOutput* dropout_fc_back = dropout_backward(model->dropout_fc, forward_result->dropout_fc_result, fc2_input_grad);
    Tensor* dropout_fc_input_grad = dropout_fc_back->input_grad;
    dropout_fc_back->input_grad = NULL;
    dropout_backward_output_free(dropout_fc_back);
    check_gradients("Dropout FC backward", dropout_fc_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout_FC: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // SiLU FC1 backward
    GET_TIME(start_op);
    Tensor* silu_fc1_grad = silu_grad(forward_result->silu1_out, dropout_fc_input_grad);
    tensor_free(dropout_fc_input_grad);
    check_gradients("SiLU FC1 backward", silu_fc1_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU_FC1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // FC1 backward
    GET_TIME(start_op);
    LinearBackwardOutput* fc1_back = linear_backward(model->fc1, forward_result->fc1_result, silu_fc1_grad);
    Tensor* fc1_input_grad = fc1_back->input_grad;
    fc1_back->input_grad = NULL;
    linear_backward_output_free(fc1_back);
    tensor_free(silu_fc1_grad);
    check_gradients("FC1 backward", fc1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - FC1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Reshape fc1 input gradient back to original conv shape [batch_size, 256, 1, 1]
    GET_TIME(start_op);
    int reshape_shape[] = {fc1_input_grad->shape[0], 256, 1, 1};
    Tensor* reshaped_fc1_grad = tensor_create(reshape_shape, 4);
    if (!reshaped_fc1_grad) {
        fprintf(stderr, "Failed to reshape fc1 gradient\n");
        return NULL;
    }
    memcpy(reshaped_fc1_grad->data, fc1_input_grad->data, fc1_input_grad->size * sizeof(float));
    check_gradients("Reshaped FC1 backward", reshaped_fc1_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Reshape: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // GAP backward
    GET_TIME(start_op);
    AdaptiveAvgPool2DBackwardOutput* gap_back = adaptive_avg_pool2d_backward(model->gap, forward_result->gap_result, reshaped_fc1_grad);
    Tensor* gap_input_grad = gap_back->input_grad;
    gap_back->input_grad = NULL;
    adaptive_avg_pool2d_backward_output_free(gap_back);
    tensor_free(reshaped_fc1_grad);
    check_gradients("GAP backward", gap_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - GAP: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // SiLU residual3 backward
    GET_TIME(start_op);
    Tensor* silu3_residual_grad = silu_grad(forward_result->silu3_residual, gap_input_grad);
    tensor_free(gap_input_grad);
    check_gradients("SiLU residual3 backward", silu3_residual_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU_residual3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual3 split: gradient goes to both main path and shortcut
    GET_TIME(start_op);
    // Main path gets the full gradient (since it's added to shortcut)
    Tensor* residual3_main_grad = tensor_create(silu3_residual_grad->shape, silu3_residual_grad->ndim);
    memcpy(residual3_main_grad->data, silu3_residual_grad->data, silu3_residual_grad->size * sizeof(float));
    // Shortcut path gets the full gradient (since it's added to main)
    Tensor* residual3_shortcut_grad = tensor_create(silu3_residual_grad->shape, silu3_residual_grad->ndim);
    memcpy(residual3_shortcut_grad->data, silu3_residual_grad->data, silu3_residual_grad->size * sizeof(float));
    // Free the original gradient tensor after copying
    tensor_free(silu3_residual_grad);
    check_gradients("Residual3 split - main", residual3_main_grad);
    check_gradients("Residual3 split - shortcut", residual3_shortcut_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Residual3 split: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Shortcut3 backward
    GET_TIME(start_op);
    Tensor* shortcut3_input_grad = conv2D_backward(model->shortcut3, forward_result->dropout2_out, residual3_shortcut_grad);
    tensor_free(residual3_shortcut_grad);
    check_gradients("Shortcut3 backward", shortcut3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Shortcut3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path backward: BN3_1x1 -> Conv3_1x1 -> SiLU3_3x3 -> BN3_3x3 -> Conv3_3x3
    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn3_1x1_back = batch_norm2d_backward(model->bn3_1x1, forward_result->bn3_1x1_result, residual3_main_grad);
    if (!bn3_1x1_back) {
        fprintf(stderr, "BN3_1x1 backward failed\n");
        return NULL;
    }
    Tensor* bn3_1x1_input_grad = bn3_1x1_back->input_grad;
    bn3_1x1_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn3_1x1_back);
    tensor_free(residual3_main_grad);
    check_gradients("BN3_1x1 backward", bn3_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN3_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv3_1x1_input_grad = conv2D_backward(model->conv3_1x1, forward_result->silu3_3x3, bn3_1x1_input_grad);
    tensor_free(bn3_1x1_input_grad);
    check_gradients("Conv3_1x1 backward", conv3_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv3_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* silu3_3x3_grad = silu_grad(forward_result->silu3_3x3, conv3_1x1_input_grad);
    tensor_free(conv3_1x1_input_grad);
    check_gradients("SiLU3_3x3 backward", silu3_3x3_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU3_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn3_3x3_back = batch_norm2d_backward(model->bn3_3x3, forward_result->bn3_3x3_result, silu3_3x3_grad);
    if (!bn3_3x3_back) {
        fprintf(stderr, "BN3_3x3 backward failed\n");
        return NULL;
    }
    Tensor* bn3_3x3_input_grad = bn3_3x3_back->input_grad;
    bn3_3x3_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn3_3x3_back);
    tensor_free(silu3_3x3_grad);
    check_gradients("BN3_3x3 backward", bn3_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN3_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv3_3x3_input_grad = conv2D_backward(model->conv3_3x3, forward_result->dropout2_out, bn3_3x3_input_grad);
    tensor_free(bn3_3x3_input_grad);
    check_gradients("Conv3_3x3 backward", conv3_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv3_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Combine gradients from shortcut and main path
    GET_TIME(start_op);
    Tensor* block3_input_grad = tensor_add(conv3_3x3_input_grad, shortcut3_input_grad);
    tensor_free(conv3_3x3_input_grad);
    tensor_free(shortcut3_input_grad);
    check_gradients("Block3 combined gradient", block3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Block3 combine: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Dropout2 backward
    GET_TIME(start_op);
    Dropout2DBackwardOutput* dropout2_back = dropout2d_backward(model->dropout_conv, forward_result->dropout2_result, block3_input_grad);
    Tensor* dropout2_input_grad = dropout2_back->input_grad;
    dropout2_back->input_grad = NULL;
    dropout2d_backward_output_free(dropout2_back);
    tensor_free(block3_input_grad);
    check_gradients("Dropout2 backward", dropout2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Pool2 backward
    GET_TIME(start_op);
    MaxPool2DBackwardOutput* pool2_back = maxpool2d_backward(model->pool2, forward_result->pool2_result, dropout2_input_grad);
    Tensor* pool2_input_grad = pool2_back->input_grad;
    pool2_back->input_grad = NULL;
    maxpool2d_backward_output_free(pool2_back);
    tensor_free(dropout2_input_grad);
    check_gradients("Pool2 backward", pool2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Pool2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // SiLU residual2 backward
    GET_TIME(start_op);
    Tensor* silu2_residual_grad = silu_grad(forward_result->silu2_residual, pool2_input_grad);
    tensor_free(pool2_input_grad);
    check_gradients("SiLU residual2 backward", silu2_residual_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU_residual2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual2 split
    GET_TIME(start_op);
    Tensor* residual2_main_grad = tensor_create(silu2_residual_grad->shape, silu2_residual_grad->ndim);
    memcpy(residual2_main_grad->data, silu2_residual_grad->data, silu2_residual_grad->size * sizeof(float));
    Tensor* residual2_shortcut_grad = tensor_create(silu2_residual_grad->shape, silu2_residual_grad->ndim);
    memcpy(residual2_shortcut_grad->data, silu2_residual_grad->data, silu2_residual_grad->size * sizeof(float));
    // Free the original gradient tensor after copying
    tensor_free(silu2_residual_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Residual2 split: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Shortcut2 backward
    GET_TIME(start_op);
    Tensor* shortcut2_input_grad = conv2D_backward(model->shortcut2, forward_result->dropout1_out, residual2_shortcut_grad);
    tensor_free(residual2_shortcut_grad);
    check_gradients("Shortcut2 backward", shortcut2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Shortcut2: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path backward: BN2_1x1 -> Conv2_1x1 -> SiLU2_3x3 -> BN2_3x3 -> Conv2_3x3
    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn2_1x1_back = batch_norm2d_backward(model->bn2_1x1, forward_result->bn2_1x1_result, residual2_main_grad);
    if (!bn2_1x1_back) {
        fprintf(stderr, "BN2_1x1 backward failed\n");
        return NULL;
    }
    Tensor* bn2_1x1_input_grad = bn2_1x1_back->input_grad;
    bn2_1x1_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn2_1x1_back);
    tensor_free(residual2_main_grad);
    check_gradients("BN2_1x1 backward", bn2_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN2_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv2_1x1_input_grad = conv2D_backward(model->conv2_1x1, forward_result->silu2_3x3, bn2_1x1_input_grad);
    tensor_free(bn2_1x1_input_grad);
    check_gradients("Conv2_1x1 backward", conv2_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv2_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* silu2_3x3_grad = silu_grad(forward_result->silu2_3x3, conv2_1x1_input_grad);
    tensor_free(conv2_1x1_input_grad);
    check_gradients("SiLU2_3x3 backward", silu2_3x3_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU2_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn2_3x3_back = batch_norm2d_backward(model->bn2_3x3, forward_result->bn2_3x3_result, silu2_3x3_grad);
    if (!bn2_3x3_back) {
        fprintf(stderr, "BN2_3x3 backward failed\n");
        return NULL;
    }
    Tensor* bn2_3x3_input_grad = bn2_3x3_back->input_grad;
    bn2_3x3_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn2_3x3_back);
    tensor_free(silu2_3x3_grad);
    check_gradients("BN2_3x3 backward", bn2_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN2_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv2_3x3_input_grad = conv2D_backward(model->conv2_3x3, forward_result->dropout1_out, bn2_3x3_input_grad);
    tensor_free(bn2_3x3_input_grad);
    check_gradients("Conv2_3x3 backward", conv2_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv2_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Combine gradients from shortcut and main path
    GET_TIME(start_op);
    Tensor* block2_input_grad = tensor_add(conv2_3x3_input_grad, shortcut2_input_grad);
    tensor_free(conv2_3x3_input_grad);
    tensor_free(shortcut2_input_grad);
    check_gradients("Block2 combined gradient", block2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Block2 combine: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Dropout1 backward
    GET_TIME(start_op);
    Dropout2DBackwardOutput* dropout1_back = dropout2d_backward(model->dropout_conv, forward_result->dropout1_result, block2_input_grad);
    Tensor* dropout1_input_grad = dropout1_back->input_grad;
    dropout1_back->input_grad = NULL;
    dropout2d_backward_output_free(dropout1_back);
    tensor_free(block2_input_grad);
    check_gradients("Dropout1 backward", dropout1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Pool1 backward
    GET_TIME(start_op);
    MaxPool2DBackwardOutput* pool1_back = maxpool2d_backward(model->pool1, forward_result->pool1_result, dropout1_input_grad);
    Tensor* pool1_input_grad = pool1_back->input_grad;
    pool1_back->input_grad = NULL;
    maxpool2d_backward_output_free(pool1_back);
    tensor_free(dropout1_input_grad);
    check_gradients("Pool1 backward", pool1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Pool1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // SiLU residual1 backward
    GET_TIME(start_op);
    Tensor* silu1_residual_grad = silu_grad(forward_result->silu1_residual, pool1_input_grad);
    tensor_free(pool1_input_grad);
    check_gradients("SiLU residual1 backward", silu1_residual_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU_residual1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Residual1 split
    GET_TIME(start_op);
    Tensor* residual1_main_grad = tensor_create(silu1_residual_grad->shape, silu1_residual_grad->ndim);
    memcpy(residual1_main_grad->data, silu1_residual_grad->data, silu1_residual_grad->size * sizeof(float));
    Tensor* residual1_shortcut_grad = tensor_create(silu1_residual_grad->shape, silu1_residual_grad->ndim);
    memcpy(residual1_shortcut_grad->data, silu1_residual_grad->data, silu1_residual_grad->size * sizeof(float));
    // Free the original gradient tensor after copying
    tensor_free(silu1_residual_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Residual1 split: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Shortcut1 backward
    GET_TIME(start_op);
    Tensor* shortcut1_input_grad = conv2D_backward(model->shortcut1, forward_result->input, residual1_shortcut_grad);
    tensor_free(residual1_shortcut_grad);
    check_gradients("Shortcut1 backward", shortcut1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Shortcut1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Main path backward: BN1_1x1 -> Conv1_1x1 -> SiLU1_3x3 -> BN1_3x3 -> Conv1_3x3
    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn1_1x1_back = batch_norm2d_backward(model->bn1_1x1, forward_result->bn1_1x1_result, residual1_main_grad);
    if (!bn1_1x1_back) {
        fprintf(stderr, "BN1_1x1 backward failed\n");
        return NULL;
    }
    Tensor* bn1_1x1_input_grad = bn1_1x1_back->input_grad;
    bn1_1x1_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn1_1x1_back);
    tensor_free(residual1_main_grad);
    check_gradients("BN1_1x1 backward", bn1_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN1_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv1_1x1_input_grad = conv2D_backward(model->conv1_1x1, forward_result->silu1_3x3, bn1_1x1_input_grad);
    tensor_free(bn1_1x1_input_grad);
    check_gradients("Conv1_1x1 backward", conv1_1x1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv1_1x1: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* silu1_3x3_grad = silu_grad(forward_result->silu1_3x3, conv1_1x1_input_grad);
    tensor_free(conv1_1x1_input_grad);
    check_gradients("SiLU1_3x3 backward", silu1_3x3_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - SiLU1_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    BatchNorm2DBackwardOutput* bn1_3x3_back = batch_norm2d_backward(model->bn1_3x3, forward_result->bn1_3x3_result, silu1_3x3_grad);
    if (!bn1_3x3_back) {
        fprintf(stderr, "BN1_3x3 backward failed\n");
        return NULL;
    }
    Tensor* bn1_3x3_input_grad = bn1_3x3_back->input_grad;
    bn1_3x3_back->input_grad = NULL;
    batch_norm2d_backward_output_free(bn1_3x3_back);
    tensor_free(silu1_3x3_grad);
    check_gradients("BN1_3x3 backward", bn1_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - BN1_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(start_op);
    Tensor* conv1_3x3_input_grad = conv2D_backward(model->conv1_3x3, forward_result->input, bn1_3x3_input_grad);
    if (!conv1_3x3_input_grad) {
        fprintf(stderr, "Conv1_3x3 backward failed\n");
        tensor_free(bn1_3x3_input_grad);
        return NULL;
    }
    tensor_free(bn1_3x3_input_grad);
    check_gradients("Conv1_3x3 backward", conv1_3x3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv1_3x3: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    // Combine gradients from shortcut and main path
    GET_TIME(start_op);
    Tensor* input_grad = tensor_add(conv1_3x3_input_grad, shortcut1_input_grad);
    if (!input_grad) {
        fprintf(stderr, "Input gradient tensor_add failed\n");
        tensor_free(conv1_3x3_input_grad);
        tensor_free(shortcut1_input_grad);
        return NULL;
    }
    tensor_free(conv1_3x3_input_grad);
    tensor_free(shortcut1_input_grad);
    check_gradients("Input gradient", input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Input combine: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Backward - Total: %.3f ms\n", TIME_DIFF(start_total, end_total));
    }

    return input_grad;
}

// Training step (loss + backward + optimizer step)
float cnn_training_step(CNN* model, CNNForwardResult* forward_result, Tensor* target) {
    TimingType start_total, end_total, start_op, end_op;
    GET_TIME(start_total);

    static int forward_debug_count = 0;
    if (forward_debug_count++ < 3) {
        check_gradients("Training step input", forward_result->input);
        check_gradients("Training step FC2 output", forward_result->fc2_out);
    }

    GET_TIME(start_op);
    CrossEntropyOutput* loss_result = cross_entropy_loss_forward(model->criterion, forward_result->fc2_out, target);
    if (!loss_result) {
        fprintf(stderr, "Loss computation failed\n");
        return 0.0f;
    }
    float loss = loss_result->loss;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Training Step - Loss Computation: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    adam_zero_grad(model->optimizer);
    GET_TIME(start_op);
    Tensor* input_grad = cnn_backward(model, forward_result, loss_result);
    if (!input_grad) {
        fprintf(stderr, "Backward pass failed\n");
        cross_entropy_result_free(loss_result);
        return 0.0f;
    }
    tensor_free(input_grad);
    GET_TIME(end_op);


    GET_TIME(start_op);

    adam_step(model->optimizer);
    GET_TIME(end_op);

    if (model->timing_verbose) {
        printf("Training Step - Optimizer Step: %.3f ms\n", TIME_DIFF(start_op, end_op));
    }

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Training Step - Total: %.3f ms\n", TIME_DIFF(start_total, end_total));
    }

    cross_entropy_result_free(loss_result);
    return loss;
}

// Inference (returns predicted class indices)
Tensor* cnn_predict(CNN* model, Tensor* input) {
    CNNForwardResult* forward_result = cnn_forward(model, input);

    int pred_shape[] = {input->shape[0], 1, 1, 1};
    Tensor* predictions = tensor_create(pred_shape, 4);

    for (int b = 0; b < input->shape[0]; b++) {
        int max_idx = 0;
        float max_val = forward_result->fc2_out->data[b * 26];

        for (int c = 1; c < 26; c++) {
            float val = forward_result->fc2_out->data[b * 26 + c];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        predictions->data[b] = (float)max_idx; // 0-based indexing: 0-25 for a-z
    }

    cnn_forward_result_free(forward_result);

    return predictions;
}

void cnn_step_optimizer(CNN* model) {
    if (model->optimizer) {
        adam_step(model->optimizer);
    }
}

void cnn_step_scheduler(CNN* model) {
    if (model->scheduler) {
        step_lr_step(model->scheduler);
    }
}

int cnn_get_parameters(CNN* model, Tensor*** params, Tensor*** grads) {
    int num_params = 36;  // 9 conv layers * 2 + 6 batch norms * 2 + 2 fc layers * 2 = 36

    *params = (Tensor**)malloc(num_params * sizeof(Tensor*));
    *grads = (Tensor**)malloc(num_params * sizeof(Tensor*));

    int idx = 0;

    // Block 1 convolutional layers + shortcuts
    (*params)[idx] = model->conv1_3x3->weight;
    (*grads)[idx++] = model->conv1_3x3->weight_grad;

    (*params)[idx] = model->conv1_3x3->bias;
    (*grads)[idx++] = model->conv1_3x3->bias_grad;

    (*params)[idx] = model->bn1_3x3->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn1_3x3->layer_grad->weight_grad;

    (*params)[idx] = model->bn1_3x3->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn1_3x3->layer_grad->bias_grad;

    (*params)[idx] = model->conv1_1x1->weight;
    (*grads)[idx++] = model->conv1_1x1->weight_grad;

    (*params)[idx] = model->conv1_1x1->bias;
    (*grads)[idx++] = model->conv1_1x1->bias_grad;

    (*params)[idx] = model->bn1_1x1->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn1_1x1->layer_grad->weight_grad;

    (*params)[idx] = model->bn1_1x1->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn1_1x1->layer_grad->bias_grad;

    (*params)[idx] = model->shortcut1->weight;
    (*grads)[idx++] = model->shortcut1->weight_grad;

    (*params)[idx] = model->shortcut1->bias;
    (*grads)[idx++] = model->shortcut1->bias_grad;

    // Block 2 convolutional layers + shortcuts
    (*params)[idx] = model->conv2_3x3->weight;
    (*grads)[idx++] = model->conv2_3x3->weight_grad;

    (*params)[idx] = model->conv2_3x3->bias;
    (*grads)[idx++] = model->conv2_3x3->bias_grad;

    (*params)[idx] = model->bn2_3x3->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn2_3x3->layer_grad->weight_grad;

    (*params)[idx] = model->bn2_3x3->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn2_3x3->layer_grad->bias_grad;

    (*params)[idx] = model->conv2_1x1->weight;
    (*grads)[idx++] = model->conv2_1x1->weight_grad;

    (*params)[idx] = model->conv2_1x1->bias;
    (*grads)[idx++] = model->conv2_1x1->bias_grad;

    (*params)[idx] = model->bn2_1x1->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn2_1x1->layer_grad->weight_grad;

    (*params)[idx] = model->bn2_1x1->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn2_1x1->layer_grad->bias_grad;

    (*params)[idx] = model->shortcut2->weight;
    (*grads)[idx++] = model->shortcut2->weight_grad;

    (*params)[idx] = model->shortcut2->bias;
    (*grads)[idx++] = model->shortcut2->bias_grad;

    // Block 3 convolutional layers + shortcuts
    (*params)[idx] = model->conv3_3x3->weight;
    (*grads)[idx++] = model->conv3_3x3->weight_grad;

    (*params)[idx] = model->conv3_3x3->bias;
    (*grads)[idx++] = model->conv3_3x3->bias_grad;

    (*params)[idx] = model->bn3_3x3->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn3_3x3->layer_grad->weight_grad;

    (*params)[idx] = model->bn3_3x3->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn3_3x3->layer_grad->bias_grad;

    (*params)[idx] = model->conv3_1x1->weight;
    (*grads)[idx++] = model->conv3_1x1->weight_grad;

    (*params)[idx] = model->conv3_1x1->bias;
    (*grads)[idx++] = model->conv3_1x1->bias_grad;

    (*params)[idx] = model->bn3_1x1->layer_grad->weights;  // gamma
    (*grads)[idx++] = model->bn3_1x1->layer_grad->weight_grad;

    (*params)[idx] = model->bn3_1x1->layer_grad->biases;   // beta
    (*grads)[idx++] = model->bn3_1x1->layer_grad->bias_grad;

    (*params)[idx] = model->shortcut3->weight;
    (*grads)[idx++] = model->shortcut3->weight_grad;

    (*params)[idx] = model->shortcut3->bias;
    (*grads)[idx++] = model->shortcut3->bias_grad;

    // Fully connected layers
    (*params)[idx] = model->fc1->layer_grad->weights;
    (*grads)[idx++] = model->fc1->layer_grad->weight_grad;

    (*params)[idx] = model->fc1->layer_grad->biases;
    (*grads)[idx++] = model->fc1->layer_grad->bias_grad;

    (*params)[idx] = model->fc2->layer_grad->weights;
    (*grads)[idx++] = model->fc2->layer_grad->weight_grad;

    (*params)[idx] = model->fc2->layer_grad->biases;
    (*grads)[idx++] = model->fc2->layer_grad->bias_grad;

    return num_params;
}

void cnn_free_parameters(Tensor** params, Tensor** grads) {
    if (params) {
        free(params);
    }
    if (grads) {
        free(grads);
    }
}

int cnn_load_weights(CNN* model, int epoch) {
    // Use larger buffer to prevent overflow (max needed is ~46 chars)
    #define PATH_BUFFER_SIZE 512
    char conv1_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv1_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn1_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn1_3x3_beta_path[PATH_BUFFER_SIZE];
    char conv1_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv1_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn1_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn1_1x1_beta_path[PATH_BUFFER_SIZE];
    char shortcut1_weight_path[PATH_BUFFER_SIZE];
    char shortcut1_bias_path[PATH_BUFFER_SIZE];

    char conv2_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv2_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn2_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn2_3x3_beta_path[PATH_BUFFER_SIZE];
    char conv2_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv2_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn2_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn2_1x1_beta_path[PATH_BUFFER_SIZE];
    char shortcut2_weight_path[PATH_BUFFER_SIZE];
    char shortcut2_bias_path[PATH_BUFFER_SIZE];

    char conv3_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv3_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn3_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn3_3x3_beta_path[PATH_BUFFER_SIZE];
    char conv3_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv3_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn3_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn3_1x1_beta_path[PATH_BUFFER_SIZE];
    char shortcut3_weight_path[PATH_BUFFER_SIZE];
    char shortcut3_bias_path[PATH_BUFFER_SIZE];

    char fc1_weight_path[PATH_BUFFER_SIZE];
    char fc1_bias_path[PATH_BUFFER_SIZE];
    char fc2_weight_path[PATH_BUFFER_SIZE];
    char fc2_bias_path[PATH_BUFFER_SIZE];

    sprintf(conv1_3x3_weight_path, "weights/conv1_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv1_3x3_bias_path, "weights/conv1_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_gamma_path, "weights/bn1_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_beta_path, "weights/bn1_3x3_beta_epoch_%d.bin", epoch);
    sprintf(conv1_1x1_weight_path, "weights/conv1_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv1_1x1_bias_path, "weights/conv1_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_gamma_path, "weights/bn1_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_beta_path, "weights/bn1_1x1_beta_epoch_%d.bin", epoch);
    sprintf(shortcut1_weight_path, "weights/shortcut1_weight_epoch_%d.bin", epoch);
    sprintf(shortcut1_bias_path, "weights/shortcut1_bias_epoch_%d.bin", epoch);

    sprintf(conv2_3x3_weight_path, "weights/conv2_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv2_3x3_bias_path, "weights/conv2_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_gamma_path, "weights/bn2_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_beta_path, "weights/bn2_3x3_beta_epoch_%d.bin", epoch);
    sprintf(conv2_1x1_weight_path, "weights/conv2_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv2_1x1_bias_path, "weights/conv2_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_gamma_path, "weights/bn2_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_beta_path, "weights/bn2_1x1_beta_epoch_%d.bin", epoch);
    sprintf(shortcut2_weight_path, "weights/shortcut2_weight_epoch_%d.bin", epoch);
    sprintf(shortcut2_bias_path, "weights/shortcut2_bias_epoch_%d.bin", epoch);

    sprintf(conv3_3x3_weight_path, "weights/conv3_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv3_3x3_bias_path, "weights/conv3_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_gamma_path, "weights/bn3_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_beta_path, "weights/bn3_3x3_beta_epoch_%d.bin", epoch);
    sprintf(conv3_1x1_weight_path, "weights/conv3_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv3_1x1_bias_path, "weights/conv3_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_gamma_path, "weights/bn3_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_beta_path, "weights/bn3_1x1_beta_epoch_%d.bin", epoch);
    sprintf(shortcut3_weight_path, "weights/shortcut3_weight_epoch_%d.bin", epoch);
    sprintf(shortcut3_bias_path, "weights/shortcut3_bias_epoch_%d.bin", epoch);

    sprintf(fc1_weight_path, "weights/fc1_weight_epoch_%d.bin", epoch);
    sprintf(fc1_bias_path, "weights/fc1_bias_epoch_%d.bin", epoch);
    sprintf(fc2_weight_path, "weights/fc2_weight_epoch_%d.bin", epoch);
    sprintf(fc2_bias_path, "weights/fc2_bias_epoch_%d.bin", epoch);

    return cnn_load_weights_from_files(model,
                                             conv1_3x3_weight_path, conv1_3x3_bias_path,
                                             bn1_3x3_gamma_path, bn1_3x3_beta_path,
                                             conv1_1x1_weight_path, conv1_1x1_bias_path,
                                             bn1_1x1_gamma_path, bn1_1x1_beta_path,
                                             shortcut1_weight_path, shortcut1_bias_path,

                                             conv2_3x3_weight_path, conv2_3x3_bias_path,
                                             bn2_3x3_gamma_path, bn2_3x3_beta_path,
                                             conv2_1x1_weight_path, conv2_1x1_bias_path,
                                             bn2_1x1_gamma_path, bn2_1x1_beta_path,
                                             shortcut2_weight_path, shortcut2_bias_path,

                                             conv3_3x3_weight_path, conv3_3x3_bias_path,
                                             bn3_3x3_gamma_path, bn3_3x3_beta_path,
                                             conv3_1x1_weight_path, conv3_1x1_bias_path,
                                             bn3_1x1_gamma_path, bn3_1x1_beta_path,
                                             shortcut3_weight_path, shortcut3_bias_path,

                                             fc1_weight_path, fc1_bias_path,
                                             fc2_weight_path, fc2_bias_path);
}

static int load_tensor_from_file(const char* filepath, Tensor* tensor, const char* description) {
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", filepath);
        return 0;
    }

    size_t read_count = fread(tensor->data, sizeof(float), tensor->size, f);
    if (read_count != (size_t)tensor->size) {
        fprintf(stderr, "Failed to read %s from %s\n", description, filepath);
        fclose(f);
        return 0;
    }

    fclose(f);
    return 1;
}

int cnn_load_weights_from_files(CNN* model,
                                       const char* conv1_3x3_weight_path,
                                       const char* conv1_3x3_bias_path,
                                       const char* bn1_3x3_gamma_path,
                                       const char* bn1_3x3_beta_path,
                                       const char* conv1_1x1_weight_path,
                                       const char* conv1_1x1_bias_path,
                                       const char* bn1_1x1_gamma_path,
                                       const char* bn1_1x1_beta_path,
                                       const char* shortcut1_weight_path,
                                       const char* shortcut1_bias_path,

                                       const char* conv2_3x3_weight_path,
                                       const char* conv2_3x3_bias_path,
                                       const char* bn2_3x3_gamma_path,
                                       const char* bn2_3x3_beta_path,
                                       const char* conv2_1x1_weight_path,
                                       const char* conv2_1x1_bias_path,
                                       const char* bn2_1x1_gamma_path,
                                       const char* bn2_1x1_beta_path,
                                       const char* shortcut2_weight_path,
                                       const char* shortcut2_bias_path,

                                       const char* conv3_3x3_weight_path,
                                       const char* conv3_3x3_bias_path,
                                       const char* bn3_3x3_gamma_path,
                                       const char* bn3_3x3_beta_path,
                                       const char* conv3_1x1_weight_path,
                                       const char* conv3_1x1_bias_path,
                                       const char* bn3_1x1_gamma_path,
                                       const char* bn3_1x1_beta_path,
                                       const char* shortcut3_weight_path,
                                       const char* shortcut3_bias_path,

                                       const char* fc1_weight_path,
                                       const char* fc1_bias_path,
                                       const char* fc2_weight_path,
                                       const char* fc2_bias_path) {

    // Block 1
    if (!load_tensor_from_file(conv1_3x3_weight_path, model->conv1_3x3->weight, "conv1_3x3 weights")) return 0;
    if (!load_tensor_from_file(conv1_3x3_bias_path, model->conv1_3x3->bias, "conv1_3x3 bias")) return 0;
    if (!load_tensor_from_file(bn1_3x3_gamma_path, model->bn1_3x3->layer_grad->weights, "bn1_3x3 gamma")) return 0;
    if (!load_tensor_from_file(bn1_3x3_beta_path, model->bn1_3x3->layer_grad->biases, "bn1_3x3 beta")) return 0;
    if (!load_tensor_from_file(conv1_1x1_weight_path, model->conv1_1x1->weight, "conv1_1x1 weights")) return 0;
    if (!load_tensor_from_file(conv1_1x1_bias_path, model->conv1_1x1->bias, "conv1_1x1 bias")) return 0;
    if (!load_tensor_from_file(bn1_1x1_gamma_path, model->bn1_1x1->layer_grad->weights, "bn1_1x1 gamma")) return 0;
    if (!load_tensor_from_file(bn1_1x1_beta_path, model->bn1_1x1->layer_grad->biases, "bn1_1x1 beta")) return 0;
    if (!load_tensor_from_file(shortcut1_weight_path, model->shortcut1->weight, "shortcut1 weights")) return 0;
    if (!load_tensor_from_file(shortcut1_bias_path, model->shortcut1->bias, "shortcut1 bias")) return 0;

    // Block 2
    if (!load_tensor_from_file(conv2_3x3_weight_path, model->conv2_3x3->weight, "conv2_3x3 weights")) return 0;
    if (!load_tensor_from_file(conv2_3x3_bias_path, model->conv2_3x3->bias, "conv2_3x3 bias")) return 0;
    if (!load_tensor_from_file(bn2_3x3_gamma_path, model->bn2_3x3->layer_grad->weights, "bn2_3x3 gamma")) return 0;
    if (!load_tensor_from_file(bn2_3x3_beta_path, model->bn2_3x3->layer_grad->biases, "bn2_3x3 beta")) return 0;
    if (!load_tensor_from_file(conv2_1x1_weight_path, model->conv2_1x1->weight, "conv2_1x1 weights")) return 0;
    if (!load_tensor_from_file(conv2_1x1_bias_path, model->conv2_1x1->bias, "conv2_1x1 bias")) return 0;
    if (!load_tensor_from_file(bn2_1x1_gamma_path, model->bn2_1x1->layer_grad->weights, "bn2_1x1 gamma")) return 0;
    if (!load_tensor_from_file(bn2_1x1_beta_path, model->bn2_1x1->layer_grad->biases, "bn2_1x1 beta")) return 0;
    if (!load_tensor_from_file(shortcut2_weight_path, model->shortcut2->weight, "shortcut2 weights")) return 0;
    if (!load_tensor_from_file(shortcut2_bias_path, model->shortcut2->bias, "shortcut2 bias")) return 0;

    // Block 3
    if (!load_tensor_from_file(conv3_3x3_weight_path, model->conv3_3x3->weight, "conv3_3x3 weights")) return 0;
    if (!load_tensor_from_file(conv3_3x3_bias_path, model->conv3_3x3->bias, "conv3_3x3 bias")) return 0;
    if (!load_tensor_from_file(bn3_3x3_gamma_path, model->bn3_3x3->layer_grad->weights, "bn3_3x3 gamma")) return 0;
    if (!load_tensor_from_file(bn3_3x3_beta_path, model->bn3_3x3->layer_grad->biases, "bn3_3x3 beta")) return 0;
    if (!load_tensor_from_file(conv3_1x1_weight_path, model->conv3_1x1->weight, "conv3_1x1 weights")) return 0;
    if (!load_tensor_from_file(conv3_1x1_bias_path, model->conv3_1x1->bias, "conv3_1x1 bias")) return 0;
    if (!load_tensor_from_file(bn3_1x1_gamma_path, model->bn3_1x1->layer_grad->weights, "bn3_1x1 gamma")) return 0;
    if (!load_tensor_from_file(bn3_1x1_beta_path, model->bn3_1x1->layer_grad->biases, "bn3_1x1 beta")) return 0;
    if (!load_tensor_from_file(shortcut3_weight_path, model->shortcut3->weight, "shortcut3 weights")) return 0;
    if (!load_tensor_from_file(shortcut3_bias_path, model->shortcut3->bias, "shortcut3 bias")) return 0;

    // FC layers
    if (!load_tensor_from_file(fc1_weight_path, model->fc1->layer_grad->weights, "fc1 weights")) return 0;
    if (!load_tensor_from_file(fc1_bias_path, model->fc1->layer_grad->biases, "fc1 bias")) return 0;
    if (!load_tensor_from_file(fc2_weight_path, model->fc2->layer_grad->weights, "fc2 weights")) return 0;
    if (!load_tensor_from_file(fc2_bias_path, model->fc2->layer_grad->biases, "fc2 bias")) return 0;

    printf("Weights loaded successfully from files\n");
    return 1;
    #undef PATH_BUFFER_SIZE
}