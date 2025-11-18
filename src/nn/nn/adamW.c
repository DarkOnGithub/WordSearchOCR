#include "nn/nn/adamW.h"
#include "nn/core/tensor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

Adam* adam_create(float learning_rate, float beta1, float beta2,
                  float epsilon, float weight_decay) {
    Adam* optimizer = (Adam*)malloc(sizeof(Adam));
    if (!optimizer) return NULL;

    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->weight_decay = weight_decay;
    optimizer->t = 0;

    optimizer->param_states = NULL;
    optimizer->num_params = 0;
    optimizer->capacity = 0;

    return optimizer;
}

void adam_free(Adam* optimizer) {
    if (!optimizer) return;

    for (int i = 0; i < optimizer->num_params; i++) {
        if (optimizer->param_states[i].m) tensor_free(optimizer->param_states[i].m);
        if (optimizer->param_states[i].v) tensor_free(optimizer->param_states[i].v);
    }

    free(optimizer->param_states);
    free(optimizer);
}

int adam_add_param(Adam* optimizer, Tensor* param, Tensor* grad) {
    if (!optimizer || !param || !grad) return -1;

    if (optimizer->num_params >= optimizer->capacity) {
        int new_capacity = optimizer->capacity == 0 ? 4 : optimizer->capacity * 2;
        AdamParamState* new_states = (AdamParamState*)realloc(
            optimizer->param_states, new_capacity * sizeof(AdamParamState));
        if (!new_states) return -1;

        optimizer->param_states = new_states;
        optimizer->capacity = new_capacity;
    }

    optimizer->param_states[optimizer->num_params].param = param;
    optimizer->param_states[optimizer->num_params].grad = grad;

    optimizer->param_states[optimizer->num_params].m = tensor_create_zero(
        param->shape, param->ndim);
    optimizer->param_states[optimizer->num_params].v = tensor_create_zero(
        param->shape, param->ndim);

    if (!optimizer->param_states[optimizer->num_params].m ||
        !optimizer->param_states[optimizer->num_params].v) {
        if (optimizer->param_states[optimizer->num_params].m)
            tensor_free(optimizer->param_states[optimizer->num_params].m);
        if (optimizer->param_states[optimizer->num_params].v)
            tensor_free(optimizer->param_states[optimizer->num_params].v);
        return -1;
    }

    optimizer->num_params++;
    return 0;
}

void adam_step(Adam* optimizer) {
    if (!optimizer || optimizer->num_params == 0 || !optimizer->param_states) return;

    optimizer->t++;

    float beta1_t = powf(optimizer->beta1, optimizer->t);
    float beta2_t = powf(optimizer->beta2, optimizer->t);

    for (int i = 0; i < optimizer->num_params; i++) {
        AdamParamState* state = &optimizer->param_states[i];
        Tensor* param = state->param;
        Tensor* grad = state->grad;
        Tensor* m = state->m;
        Tensor* v = state->v;

        if (!param || !grad || !m || !v) continue;


        // Update m: m = beta1 * m + (1 - beta1) * effective_grad
        tensor_scale_inplace(m, optimizer->beta1);
        Tensor* grad_scaled = tensor_multiply_scalar_copy(grad, 1.0f - optimizer->beta1);
        tensor_add_inplace(m, grad_scaled);
        tensor_free(grad_scaled);

        // Update v: v = beta2 * v + (1 - beta2) * effective_grad^2
        tensor_scale_inplace(v, optimizer->beta2);
        Tensor* grad_squared = tensor_square(grad);
        tensor_scale_inplace(grad_squared, 1.0f - optimizer->beta2);
        tensor_add_inplace(v, grad_squared);
        tensor_free(grad_squared);

        // Compute bias-corrected moments: m̂ = m / (1 - beta1^t), v̂ = v / (1 - beta2^t)
        Tensor* m_hat = tensor_multiply_scalar_copy(m, 1.0f / (1.0f - beta1_t));
        Tensor* v_hat = tensor_multiply_scalar_copy(v, 1.0f / (1.0f - beta2_t));

        if (m_hat && v_hat) {
            // Compute sqrt(v̂) + epsilon
            Tensor* v_sqrt = tensor_sqrt(v_hat);
            Tensor* v_sqrt_eps = tensor_add_scalar_copy(v_sqrt, optimizer->epsilon);

            if (v_sqrt_eps) {
                //Apply decoupled weight decay directly to parameters
                if (optimizer->weight_decay > 0.0f) {
                    tensor_scale_inplace(param, 1.0f - optimizer->learning_rate * optimizer->weight_decay);
                }

                // Compute update: learning_rate * m̂ / (sqrt(v̂) + epsilon)
                // First compute m̂ / (sqrt(v̂) + epsilon), then scale by learning_rate
                Tensor* ratio = tensor_divide(m_hat, v_sqrt_eps);
                tensor_scale_inplace(ratio, optimizer->learning_rate);

                // Update parameter: param = param - update
                tensor_subtract_inplace(param, ratio);

                tensor_free(ratio);
                tensor_free(v_sqrt_eps);
            }

            tensor_free(v_sqrt);
            tensor_free(v_hat);
        }

        tensor_free(m_hat);

    }
}

void adam_zero_grad(Adam* optimizer) {
    if (!optimizer || !optimizer->param_states) return;

    for (int i = 0; i < optimizer->num_params; i++) {
        Tensor* grad = optimizer->param_states[i].grad;
        if (grad) {
            memset(grad->data, 0, grad->size * sizeof(float));
        }
    }
}

float adam_get_learning_rate(Adam* optimizer) {
    return optimizer ? optimizer->learning_rate : 0.0f;
}

void adam_set_learning_rate(Adam* optimizer, float lr) {
    if (optimizer) optimizer->learning_rate = lr;
}

int adam_get_num_params(Adam* optimizer) {
    return optimizer ? optimizer->num_params : 0;
}

float adam_get_beta1(Adam* optimizer) {
    return optimizer ? optimizer->beta1 : 0.0f;
}

float adam_get_beta2(Adam* optimizer) {
    return optimizer ? optimizer->beta2 : 0.0f;
}

float adam_get_epsilon(Adam* optimizer) {
    return optimizer ? optimizer->epsilon : 0.0f;
}

float adam_get_weight_decay(Adam* optimizer) {
    return optimizer ? optimizer->weight_decay : 0.0f;
}

int adam_get_t(Adam* optimizer) {
    return optimizer ? optimizer->t : 0;
}

