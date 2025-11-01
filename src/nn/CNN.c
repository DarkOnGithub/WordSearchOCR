#define _POSIX_C_SOURCE 199309L
#include "../include/nn/cnn.h"
#include "../include/nn/nn/relu.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef struct timespec TimingType;
#define GET_TIME(t) clock_gettime(CLOCK_MONOTONIC, &(t))
#define TIME_DIFF(start, end, freq) ((double)((end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec)) / 1000000.0) // ms
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

    model->conv1 = conv2D_create(1, 32, 3, 1, 1);      // 1->32, 3x3, stride=1, padding=1
    model->conv2 = conv2D_create(32, 64, 3, 1, 1);     // 32->64, 3x3, stride=1, padding=1
    model->conv3 = conv2D_create(64, 128, 3, 1, 1);    // 64->128, 3x3, stride=1, padding=1

    model->pool1 = maxpool2d_create_simple(2, 2);       // 2x2 maxpool
    model->pool2 = maxpool2d_create_simple(2, 2);       // 2x2 maxpool
    model->pool3 = maxpool2d_create_simple(2, 2);       // 2x2 maxpool

    model->dropout_conv1 = dropout2d_create(0.25f);
    model->dropout_conv2 = dropout2d_create(0.25f);
    model->dropout_conv3 = dropout2d_create(0.25f);
    model->dropout_fc = dropout_create(0.5f);

    // After 3 conv+pool blocks: 28->14->7->3, so 128*3*3 = 1152 -> 256 -> 26
    model->fc1 = linear_create(128 * 3 * 3, 256);
    model->fc2 = linear_create(256, 26);

    model->optimizer = NULL;
    model->scheduler = NULL;
    model->criterion = cross_entropy_loss_create();

    model->training = true;

    model->timing_verbose = false;

    return model;
}

void cnn_free(CNN* model) {
    if (!model) return;

    conv2D_free(model->conv1);
    conv2D_free(model->conv2);
    conv2D_free(model->conv3);

    maxpool2d_free(model->pool1);
    maxpool2d_free(model->pool2);
    maxpool2d_free(model->pool3);

    dropout2d_free(model->dropout_conv1);
    dropout2d_free(model->dropout_conv2);
    dropout2d_free(model->dropout_conv3);
    dropout_free(model->dropout_fc);

    linear_free(model->fc1);
    linear_free(model->fc2);

    if (model->optimizer) adam_free(model->optimizer);
    if (model->scheduler) step_lr_free(model->scheduler);
    cross_entropy_loss_free(model->criterion);

    free(model);
}

void cnn_train(CNN* model) {
    model->training = true;
    model->dropout_conv1->training = true;
    model->dropout_conv2->training = true;
    model->dropout_conv3->training = true;
    model->dropout_fc->training = true;
}

void cnn_eval(CNN* model) {
    model->training = false;
    model->dropout_conv1->training = false;
    model->dropout_conv2->training = false;
    model->dropout_conv3->training = false;
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

    GET_TIME(start_op);
    result->conv1_out = conv2D_forward(model->conv1, input);
    GET_TIME(end_op);
    check_gradients("Forward - Conv1 output", result->conv1_out);
    if (model->timing_verbose) {
        printf("Forward - Conv1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    result->relu1 = relu(result->conv1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - ReLU1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DOutput* pool1_result = maxpool2d_forward(model->pool1, result->relu1);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Pool1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->pool1_result = pool1_result;
    result->pool1_out = pool1_result->output;

    GET_TIME(start_op);
    Dropout2DOutput* dropout1_result = dropout2d_forward(model->dropout_conv1, result->pool1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->dropout1_result = dropout1_result;
    result->dropout1_out = dropout1_result->output;

    GET_TIME(start_op);
    result->conv2_out = conv2D_forward(model->conv2, result->dropout1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    result->relu2 = relu(result->conv2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - ReLU2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DOutput* pool2_result = maxpool2d_forward(model->pool2, result->relu2);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Pool2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->pool2_result = pool2_result;
    result->pool2_out = pool2_result->output;

    GET_TIME(start_op);
    Dropout2DOutput* dropout2_result = dropout2d_forward(model->dropout_conv2, result->pool2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->dropout2_result = dropout2_result;
    result->dropout2_out = dropout2_result->output;

    GET_TIME(start_op);
    result->conv3_out = conv2D_forward(model->conv3, result->dropout2_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Conv3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    result->relu3 = relu(result->conv3_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - ReLU3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DOutput* pool3_result = maxpool2d_forward(model->pool3, result->relu3);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Pool3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->pool3_result = pool3_result;
    result->pool3_out = pool3_result->output;

    GET_TIME(start_op);
    Dropout2DOutput* dropout3_result = dropout2d_forward(model->dropout_conv3, result->pool3_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->dropout3_result = dropout3_result;
    result->dropout3_out = dropout3_result->output;

    GET_TIME(start_op);
    int flattened_shape[] = {result->dropout3_out->shape[0],
                             result->dropout3_out->shape[1] * result->dropout3_out->shape[2] * result->dropout3_out->shape[3]};
    Tensor* flattened_copy = tensor_create(flattened_shape, 2);
    memcpy(flattened_copy->data, result->dropout3_out->data, result->dropout3_out->size * sizeof(float));
    result->flattened = flattened_copy;
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Flatten: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    LinearOutput* fc1_result = linear_forward(model->fc1, result->flattened);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - FC1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->fc1_result = fc1_result;
    result->fc1_out = fc1_result->output;

    GET_TIME(start_op);
    result->relu1_out = relu(result->fc1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - ReLU_FC1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    DropoutOutput* dropout_fc_result = dropout_forward(model->dropout_fc, result->relu1_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - Dropout_FC: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->dropout_fc_result = dropout_fc_result;
    result->dropout_fc_out = dropout_fc_result->output;

    GET_TIME(start_op);
    LinearOutput* fc2_result = linear_forward(model->fc2, result->dropout_fc_out);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Forward - FC2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }
    result->fc2_result = fc2_result;
    result->fc2_out = fc2_result->output;
    check_gradients("Forward - FC2 output", result->fc2_out);

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Forward - Total: %.3f ms\n", TIME_DIFF(start_total, end_total, timing_freq));
    }

    return result;
}

void cnn_forward_result_free(CNNForwardResult* result) {
    if (!result) return;

    tensor_free(result->conv1_out);
    tensor_free(result->relu1);
    maxpool2d_output_free(result->pool1_result);
    dropout2d_output_free(result->dropout1_result);
    tensor_free(result->conv2_out);
    tensor_free(result->relu2);
    maxpool2d_output_free(result->pool2_result);
    dropout2d_output_free(result->dropout2_result);
    tensor_free(result->conv3_out);
    tensor_free(result->relu3);
    maxpool2d_output_free(result->pool3_result);
    dropout2d_output_free(result->dropout3_result);
    tensor_free(result->flattened);
    if (result->fc1_result) {
        result->fc1_result->output = NULL;
        linear_output_free(result->fc1_result);
    }
    tensor_free(result->relu1_out);
    if (result->dropout_fc_result) {
        result->dropout_fc_result->output = NULL;
        dropout_output_free(result->dropout_fc_result);
    }
    if (result->fc2_result) {
        result->fc2_result->output = NULL;
        linear_output_free(result->fc2_result);
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
    tensor_free(loss_upstream_grad);
    check_gradients("Loss backward", output_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Loss: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    LinearBackwardOutput* fc2_back = linear_backward(model->fc2, forward_result->fc2_result, output_grad);
    Tensor* fc2_input_grad = fc2_back->input_grad;
    fc2_back->input_grad = NULL;
    linear_backward_output_free(fc2_back);
    check_gradients("FC2 backward", fc2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - FC2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    DropoutBackwardOutput* dropout_fc_back = dropout_backward(model->dropout_fc, forward_result->dropout_fc_result, fc2_input_grad);
    Tensor* dropout_fc_input_grad = dropout_fc_back->input_grad;
    dropout_fc_back->input_grad = NULL;
    dropout_backward_output_free(dropout_fc_back);
    check_gradients("Dropout FC backward", dropout_fc_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout_FC: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* relu1_grad = relu_grad(forward_result->relu1_out, dropout_fc_input_grad);
    tensor_free(dropout_fc_input_grad);
    check_gradients("ReLU FC1 backward", relu1_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - ReLU_FC1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    LinearBackwardOutput* fc1_back = linear_backward(model->fc1, forward_result->fc1_result, relu1_grad);
    Tensor* fc1_input_grad = fc1_back->input_grad;
    fc1_back->input_grad = NULL;
    linear_backward_output_free(fc1_back);
    tensor_free(relu1_grad);
    check_gradients("FC1 backward", fc1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - FC1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    // Reshape fc1 input gradient back to original conv shape [batch_size, 128, 3, 3]
    GET_TIME(start_op);
    int reshape_shape[] = {fc1_input_grad->shape[0], 128, 3, 3};
    Tensor* reshaped_fc1_grad = tensor_create(reshape_shape, 4);
    if (!reshaped_fc1_grad) {
        fprintf(stderr, "Failed to reshape fc1 gradient\n");
        return NULL;
    }
    memcpy(reshaped_fc1_grad->data, fc1_input_grad->data, fc1_input_grad->size * sizeof(float));
    check_gradients("Reshaped FC1 backward", reshaped_fc1_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Reshape: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Dropout2DBackwardOutput* dropout3_back = dropout2d_backward(model->dropout_conv3, forward_result->dropout3_result, reshaped_fc1_grad);
    if (!dropout3_back) {
        fprintf(stderr, "dropout2d_backward failed\n");
        return NULL;
    }
    Tensor* dropout3_input_grad = dropout3_back->input_grad;
    dropout3_back->input_grad = NULL;
    dropout2d_backward_output_free(dropout3_back);
    tensor_free(reshaped_fc1_grad);
    check_gradients("Dropout3 backward", dropout3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DBackwardOutput* pool3_back = maxpool2d_backward(model->pool3, forward_result->pool3_result, dropout3_input_grad);
    Tensor* pool3_input_grad = pool3_back->input_grad;
    pool3_back->input_grad = NULL;
    maxpool2d_backward_output_free(pool3_back);
    tensor_free(dropout3_input_grad);
    check_gradients("Pool3 backward", pool3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Pool3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* relu3_grad = relu_grad(forward_result->relu3, pool3_input_grad);
    tensor_free(pool3_input_grad);
    check_gradients("ReLU3 backward", relu3_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - ReLU3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* conv3_input_grad = conv2D_backward(model->conv3, forward_result->dropout2_out, relu3_grad);
    tensor_free(relu3_grad);
    check_gradients("Conv3 backward", conv3_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv3: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Dropout2DBackwardOutput* dropout2_back = dropout2d_backward(model->dropout_conv2, forward_result->dropout2_result, conv3_input_grad);
    Tensor* dropout2_input_grad = dropout2_back->input_grad;
    dropout2_back->input_grad = NULL;
    dropout2d_backward_output_free(dropout2_back);
    tensor_free(conv3_input_grad);
    check_gradients("Dropout2 backward", dropout2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DBackwardOutput* pool2_back = maxpool2d_backward(model->pool2, forward_result->pool2_result, dropout2_input_grad);
    Tensor* pool2_input_grad = pool2_back->input_grad;
    pool2_back->input_grad = NULL;
    maxpool2d_backward_output_free(pool2_back);
    tensor_free(dropout2_input_grad);
    check_gradients("Pool2 backward", pool2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Pool2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* relu2_grad = relu_grad(forward_result->relu2, pool2_input_grad);
    tensor_free(pool2_input_grad);
    check_gradients("ReLU2 backward", relu2_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - ReLU2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* conv2_input_grad = conv2D_backward(model->conv2, forward_result->dropout1_out, relu2_grad);
    tensor_free(relu2_grad);
    check_gradients("Conv2 backward", conv2_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv2: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Dropout2DBackwardOutput* dropout1_back = dropout2d_backward(model->dropout_conv1, forward_result->dropout1_result, conv2_input_grad);
    Tensor* dropout1_input_grad = dropout1_back->input_grad;
    dropout1_back->input_grad = NULL;
    dropout2d_backward_output_free(dropout1_back);
    tensor_free(conv2_input_grad);
    check_gradients("Dropout1 backward", dropout1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Dropout1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    MaxPool2DBackwardOutput* pool1_back = maxpool2d_backward(model->pool1, forward_result->pool1_result, dropout1_input_grad);
    Tensor* pool1_input_grad = pool1_back->input_grad;
    pool1_back->input_grad = NULL;
    maxpool2d_backward_output_free(pool1_back);
    tensor_free(dropout1_input_grad);
    check_gradients("Pool1 backward", pool1_input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Pool1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* relu1_grad_final = relu_grad(forward_result->relu1, pool1_input_grad);
    tensor_free(pool1_input_grad);
    check_gradients("ReLU1 backward", relu1_grad_final);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - ReLU1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(start_op);
    Tensor* input_grad = conv2D_backward(model->conv1, forward_result->input, relu1_grad_final);
    tensor_free(relu1_grad_final);
    check_gradients("Conv1 backward", input_grad);
    GET_TIME(end_op);
    if (model->timing_verbose) {
        printf("Backward - Conv1: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Backward - Total: %.3f ms\n", TIME_DIFF(start_total, end_total, timing_freq));
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
        printf("Training Step - Loss Computation: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    adam_zero_grad(model->optimizer);

    GET_TIME(start_op);
    Tensor* input_grad = cnn_backward(model, forward_result, loss_result);
    if (input_grad) tensor_free(input_grad);
    GET_TIME(end_op);


    GET_TIME(start_op);

    adam_step(model->optimizer);
    GET_TIME(end_op);

    if (model->timing_verbose) {
        printf("Training Step - Optimizer Step: %.3f ms\n", TIME_DIFF(start_op, end_op, timing_freq));
    }

    GET_TIME(end_total);
    if (model->timing_verbose) {
        printf("Training Step - Total: %.3f ms\n", TIME_DIFF(start_total, end_total, timing_freq));
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

        predictions->data[b] = (float)max_idx;
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
    int num_params = 10;

    *params = (Tensor**)malloc(num_params * sizeof(Tensor*));
    *grads = (Tensor**)malloc(num_params * sizeof(Tensor*));

    int idx = 0;

    (*params)[idx] = model->conv1->weight;
    (*grads)[idx++] = model->conv1->weight_grad;

    (*params)[idx] = model->conv1->bias;
    (*grads)[idx++] = model->conv1->bias_grad;

    (*params)[idx] = model->conv2->weight;
    (*grads)[idx++] = model->conv2->weight_grad;

    (*params)[idx] = model->conv2->bias;
    (*grads)[idx++] = model->conv2->bias_grad;

    (*params)[idx] = model->conv3->weight;
    (*grads)[idx++] = model->conv3->weight_grad;

    (*params)[idx] = model->conv3->bias;
    (*grads)[idx++] = model->conv3->bias_grad;

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

int cnn_load_weights(CNN* model, int epoch) {
    char conv1_weight_path[256];
    char conv1_bias_path[256];
    char conv2_weight_path[256];
    char conv2_bias_path[256];
    char conv3_weight_path[256];
    char conv3_bias_path[256];
    char fc1_weight_path[256];
    char fc1_bias_path[256];
    char fc2_weight_path[256];
    char fc2_bias_path[256];

    sprintf(conv1_weight_path, "weights/conv1_weight_epoch_%d.bin", epoch);
    sprintf(conv1_bias_path, "weights/conv1_bias_epoch_%d.bin", epoch);
    sprintf(conv2_weight_path, "weights/conv2_weight_epoch_%d.bin", epoch);
    sprintf(conv2_bias_path, "weights/conv2_bias_epoch_%d.bin", epoch);
    sprintf(conv3_weight_path, "weights/conv3_weight_epoch_%d.bin", epoch);
    sprintf(conv3_bias_path, "weights/conv3_bias_epoch_%d.bin", epoch);
    sprintf(fc1_weight_path, "weights/fc1_weight_epoch_%d.bin", epoch);
    sprintf(fc1_bias_path, "weights/fc1_bias_epoch_%d.bin", epoch);
    sprintf(fc2_weight_path, "weights/fc2_weight_epoch_%d.bin", epoch);
    sprintf(fc2_bias_path, "weights/fc2_bias_epoch_%d.bin", epoch);

    return cnn_load_weights_from_files(model,
                                             conv1_weight_path, conv1_bias_path,
                                             conv2_weight_path, conv2_bias_path,
                                             conv3_weight_path, conv3_bias_path,
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
                                       const char* conv1_weight_path,
                                       const char* conv1_bias_path,
                                       const char* conv2_weight_path,
                                       const char* conv2_bias_path,
                                       const char* conv3_weight_path,
                                       const char* conv3_bias_path,
                                       const char* fc1_weight_path,
                                       const char* fc1_bias_path,
                                       const char* fc2_weight_path,
                                       const char* fc2_bias_path) {

    if (!load_tensor_from_file(conv1_weight_path, model->conv1->weight, "conv1 weights")) return 0;
    if (!load_tensor_from_file(conv1_bias_path, model->conv1->bias, "conv1 bias")) return 0;

    if (!load_tensor_from_file(conv2_weight_path, model->conv2->weight, "conv2 weights")) return 0;
    if (!load_tensor_from_file(conv2_bias_path, model->conv2->bias, "conv2 bias")) return 0;

    if (!load_tensor_from_file(conv3_weight_path, model->conv3->weight, "conv3 weights")) return 0;
    if (!load_tensor_from_file(conv3_bias_path, model->conv3->bias, "conv3 bias")) return 0;

    if (!load_tensor_from_file(fc1_weight_path, model->fc1->layer_grad->weights, "fc1 weights")) return 0;
    if (!load_tensor_from_file(fc1_bias_path, model->fc1->layer_grad->biases, "fc1 bias")) return 0;

    if (!load_tensor_from_file(fc2_weight_path, model->fc2->layer_grad->weights, "fc2 weights")) return 0;
    if (!load_tensor_from_file(fc2_bias_path, model->fc2->layer_grad->biases, "fc2 bias")) return 0;

    printf("Weights loaded successfully from files\n");
    return 1;
}