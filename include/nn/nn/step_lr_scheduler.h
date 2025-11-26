#pragma once

#include "nn/nn/adamW.h"

typedef struct {
    Adam* optimizer;
    int step_size;
    float gamma;
    int last_epoch;
    float initial_lr;
} StepLR;

StepLR* step_lr_create(Adam* optimizer, int step_size, float gamma);
void step_lr_free(StepLR* scheduler);
void step_lr_step(StepLR* scheduler);
float step_lr_get_lr(StepLR* scheduler);
int step_lr_get_last_epoch(StepLR* scheduler);
void step_lr_set_last_epoch(StepLR* scheduler, int epoch);

