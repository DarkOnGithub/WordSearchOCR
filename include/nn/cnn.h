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
    Tensor* conv1_out;
    Tensor* relu1;
    MaxPool2DOutput* pool1_result;
    Tensor* pool1_out;
    Dropout2DOutput* dropout1_result;
    Tensor* dropout1_out;
    Tensor* conv2_out;
    Tensor* relu2;
    MaxPool2DOutput* pool2_result;
    Tensor* pool2_out;
    Dropout2DOutput* dropout2_result;
    Tensor* dropout2_out;
    Tensor* conv3_out;
    Tensor* relu3;
    MaxPool2DOutput* pool3_result;
    Tensor* pool3_out;
    Dropout2DOutput* dropout3_result;
    Tensor* dropout3_out;
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
    // Convolutional layers
    Conv2D* conv1;
    Conv2D* conv2;
    Conv2D* conv3;

    // Pooling layers
    MaxPool2D* pool1;
    MaxPool2D* pool2;
    MaxPool2D* pool3;

    // Dropout layers
    Dropout2D* dropout_conv1;
    Dropout2D* dropout_conv2;
    Dropout2D* dropout_conv3;
    Dropout* dropout_fc;

    // Fully connected layers
    Linear* fc1;
    Linear* fc2;

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
                               const char* conv1_weight_path,
                               const char* conv1_bias_path,
                               const char* conv2_weight_path,
                               const char* conv2_bias_path,
                               const char* conv3_weight_path,
                               const char* conv3_bias_path,
                               const char* fc1_weight_path,
                               const char* fc1_bias_path,
                               const char* fc2_weight_path,
                               const char* fc2_bias_path);

// Gradient clipping to prevent gradient explosion
void cnn_clip_gradients(CNN* model, float max_norm);
