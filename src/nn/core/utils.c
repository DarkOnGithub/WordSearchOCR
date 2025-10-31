#include "nn/core/utils.h"
#include <stdlib.h>
#include <time.h>

float random_float() {
    return (float)rand() / (float)RAND_MAX;
}

static void scale_float(float* a, float* b, float scale, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = b[i] * scale;
    }
}

static void add_float(float* a, float* b, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = a[i] + b[i];
    }
}