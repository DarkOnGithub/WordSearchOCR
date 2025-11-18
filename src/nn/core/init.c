#include "nn/core/init.h"
#include "nn/core/utils.h"
#include <math.h>

#define M_PI 3.141592653589793f

static float random_normal() {
    float u1 = random_float();
    float u2 = random_float();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void init_xavier_uniform(Tensor* tensor) {
    int fan_in, fan_out;

    if (tensor->ndim == 2) {
        fan_in = tensor->shape[1];
        fan_out = tensor->shape[0];
    } else if (tensor->ndim == 4) {
        fan_in = tensor->shape[1] * tensor->shape[2] * tensor->shape[3];
        fan_out = tensor->shape[0];
    } else {
        return;
    }

    float limit = sqrtf(6.0f / (fan_in + fan_out));

    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = (random_float() - 0.5f) * 2.0f * limit;
    }
}

void init_xavier_normal(Tensor* tensor) {
    if (tensor->ndim != 2) {
        return;
    }

    int fan_in = tensor->shape[1];
    int fan_out = tensor->shape[0];
    float std = sqrtf(2.0f / (fan_in + fan_out));

    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = random_normal() * std;
    }
}

void init_kaiming_uniform(Tensor* tensor) {
    if (tensor->ndim != 2) {
        return;
    }

    int fan_in = tensor->shape[1];
    float limit = sqrtf(6.0f / fan_in);

    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = (random_float() - 0.5f) * 2.0f * limit;
    }
}

void init_kaiming_normal(Tensor* tensor) {
    if (tensor->ndim != 2) {
        return;
    }

    int fan_in = tensor->shape[1];
    float std = sqrtf(2.0f / fan_in);

    for (int i = 0; i < tensor->size; i++) {
        tensor->data[i] = random_normal() * std;
    }
}
