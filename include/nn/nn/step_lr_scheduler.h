#pragma once

#include "nn/nn/adamW.h"

// StepLR learning rate scheduler
// Reduces learning rate by gamma every step_size epochs
typedef struct {
    Adam* optimizer;       // The optimizer whose learning rate to schedule
    int step_size;         // Number of epochs between learning rate updates
    float gamma;           // Multiplicative factor for learning rate reduction
    int last_epoch;        // Last epoch when learning rate was updated
    float initial_lr;      // Initial learning rate (for reference)
} StepLR;

// Constructor and destructor
StepLR* step_lr_create(Adam* optimizer, int step_size, float gamma);
void step_lr_free(StepLR* scheduler);

// Step the scheduler (call this after each epoch)
void step_lr_step(StepLR* scheduler);

// Get current learning rate
float step_lr_get_lr(StepLR* scheduler);

// Get last epoch
int step_lr_get_last_epoch(StepLR* scheduler);

// Manually set last epoch (useful for resuming training)
void step_lr_set_last_epoch(StepLR* scheduler, int epoch);


