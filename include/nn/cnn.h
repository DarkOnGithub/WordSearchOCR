#pragma once

#include "nn/core/tensor.h"
#include <stdbool.h>
#include "nn/layers/conv2D.h"
#include "nn/layers/maxpool2D.h"
#include "nn/layers/dropout2D.h"
#include "nn/layers/dropout.h"
#include "nn/layers/linear.h"
#include "nn/layers/cross_entropy_loss.h"
#include "nn/nn/adamW.h"
#include "nn/nn/step_lr_scheduler.h"

// Forward declaration of CNN
typedef struct CNN CNN;

// Forward result structure for CNN forward pass
typedef struct {
    Tensor* input;              // Original input

    // Block 1 outputs (3x3 -> 1x1)
    Tensor* conv1_3x3_out;
    Tensor* relu1_3x3;
    Tensor* conv1_1x1_out;
    Tensor* relu1_1x1;
    MaxPool2DOutput* pool1_result;
    Tensor* pool1_out;
    Dropout2DOutput* dropout1_result;
    Tensor* dropout1_out;

    // Block 2 outputs (3x3 -> 1x1)
    Tensor* conv2_3x3_out;
    Tensor* relu2_3x3;
    Tensor* conv2_1x1_out;
    Tensor* relu2_1x1;
    MaxPool2DOutput* pool2_result;
    Tensor* pool2_out;
    Dropout2DOutput* dropout2_result;
    Tensor* dropout2_out;

    // Block 3 outputs (3x3 -> 1x1)
    Tensor* conv3_3x3_out;
    Tensor* relu3_3x3;
    Tensor* conv3_1x1_out;
    Tensor* relu3_1x1;
    MaxPool2DOutput* pool3_result;
    Tensor* pool3_out;
    Dropout2DOutput* dropout3_result;
    Tensor* dropout3_out;

    // FC layers
    Tensor* flattened;
    LinearOutput* fc1_result;
    Tensor* fc1_out;
    Tensor* relu1_out;
    DropoutOutput* dropout_fc_result;
    Tensor* dropout_fc_out;
    LinearOutput* fc2_result;
    Tensor* fc2_out;
} CNNForwardResult;

// Main CNN structure
struct CNN {
    // Convolutional layers - Block 1 (3x3 -> 1x1)
    Conv2D* conv1_3x3;  // 1->32, 3x3
    Conv2D* conv1_1x1;  // 32->64, 1x1

    // Convolutional layers - Block 2 (3x3 -> 1x1)
    Conv2D* conv2_3x3;  // 64->64, 3x3
    Conv2D* conv2_1x1;  // 64->128, 1x1

    // Convolutional layers - Block 3 (3x3 -> 1x1)
    Conv2D* conv3_3x3;  // 128->128, 3x3
    Conv2D* conv3_1x1;  // 128->256, 1x1

    // Pooling layers
    MaxPool2D* pool1;
    MaxPool2D* pool2;
    MaxPool2D* pool3;

    // Dropout layers (hybrid rates)
    Dropout2D* dropout_conv;  // 0.20
    Dropout* dropout_fc;      // 0.45

    // Fully connected layers
    Linear* fc1;  // 256*3*3 -> 128
    Linear* fc2;  // 128 -> 26

    // Training components
    Adam* optimizer;
    StepLR* scheduler;
    CrossEntropyLoss* criterion;

    // Mode
    bool training;
    bool timing_verbose;
};

// Constructor/Destructor
CNN* cnn_create();
void cnn_free(CNN* model);

// Mode switching
void cnn_train(CNN* model);
void cnn_eval(CNN* model);

// Forward pass
CNNForwardResult* cnn_forward(CNN* model, Tensor* input);

// Free forward pass results
void cnn_forward_result_free(CNNForwardResult* result);

// Backward pass
Tensor* cnn_backward(CNN* model, CNNForwardResult* forward_result, CrossEntropyOutput* loss_result);

// Training step (loss + backward + optimizer step)
float cnn_training_step(CNN* model, CNNForwardResult* forward_result, Tensor* target);

// Inference (returns predicted class indices)
Tensor* cnn_predict(CNN* model, Tensor* input);

// Zero gradients for all parameters
void cnn_zero_grad(CNN* model);

// Step optimizer
void cnn_step_optimizer(CNN* model);

// Step scheduler
void cnn_step_scheduler(CNN* model);

// Get all parameters for optimizer setup
int cnn_get_parameters(CNN* model, Tensor*** params, Tensor*** grads);

// Load weights from files for a specific epoch
int cnn_load_weights(CNN* model, int epoch);

// Load weights from specific file paths
int cnn_load_weights_from_files(CNN* model,
                               const char* conv1_3x3_weight_path,
                               const char* conv1_3x3_bias_path,
                               const char* conv1_1x1_weight_path,
                               const char* conv1_1x1_bias_path,
                               const char* conv2_3x3_weight_path,
                               const char* conv2_3x3_bias_path,
                               const char* conv2_1x1_weight_path,
                               const char* conv2_1x1_bias_path,
                               const char* conv3_3x3_weight_path,
                               const char* conv3_3x3_bias_path,
                               const char* conv3_1x1_weight_path,
                               const char* conv3_1x1_bias_path,
                               const char* fc1_weight_path,
                               const char* fc1_bias_path,
                               const char* fc2_weight_path,
                               const char* fc2_bias_path);

// Gradient clipping to prevent gradient explosion
void cnn_clip_gradients(CNN* model, float max_norm);
