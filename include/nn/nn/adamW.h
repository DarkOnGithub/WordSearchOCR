#pragma once

#include "nn/core/tensor.h"

// Adam optimizer state for a single parameter
typedef struct {
    Tensor* param;  // Parameter tensor
    Tensor* grad;   // Gradient tensor
    Tensor* m;      // First moment estimate (momentum)
    Tensor* v;      // Second moment estimate (RMSProp)
} AdamParamState;

// Adam optimizer
typedef struct {
    float learning_rate;     // Learning rate (alpha)
    float beta1;            // Exponential decay rate for first moment (0.9)
    float beta2;            // Exponential decay rate for second moment (0.999)
    float epsilon;          // Small constant for numerical stability (1e-8)
    float weight_decay;     // L2 regularization coefficient
    float max_grad_norm;    // Maximum gradient norm for clipping (0 = disabled)
    int t;                  // Time step (iteration count)

    // Parameter states (one for each parameter tensor)
    // These will be dynamically allocated arrays
    AdamParamState* param_states;
    int num_params;
    int capacity;           // Allocated capacity for param_states array
} Adam;

// Constructor and destructor
Adam* adam_create(float learning_rate, float beta1, float beta2,
                  float epsilon, float weight_decay);
void adam_free(Adam* optimizer);

// Add parameter-gradient pair to optimizer (creates moment tensors)
int adam_add_param(Adam* optimizer, Tensor* param, Tensor* grad);

// Step function - updates all parameters based on their gradients
void adam_step(Adam* optimizer);

// Zero gradients for all parameters
void adam_zero_grad(Adam* optimizer);

// Get/set hyperparameters
float adam_get_learning_rate(Adam* optimizer);
void adam_set_learning_rate(Adam* optimizer, float lr);
void adam_set_max_grad_norm(Adam* optimizer, float max_norm);
int adam_get_num_params(Adam* optimizer);
