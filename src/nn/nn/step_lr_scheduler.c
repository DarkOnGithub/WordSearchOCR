#include "nn/nn/step_lr_scheduler.h"
#include "nn/nn/adamW.h"
#include <stdlib.h>
#include <stdio.h>

StepLR* step_lr_create(Adam* optimizer, int step_size, float gamma) {
    if (!optimizer) {
        fprintf(stderr, "Error: Optimizer cannot be NULL for StepLR scheduler\n");
        return NULL;
    }

    if (step_size <= 0) {
        fprintf(stderr, "Error: step_size must be positive for StepLR scheduler\n");
        return NULL;
    }

    if (gamma <= 0.0f || gamma > 1.0f) {
        fprintf(stderr, "Error: gamma must be in (0, 1] for StepLR scheduler\n");
        return NULL;
    }

    StepLR* scheduler = (StepLR*)malloc(sizeof(StepLR));
    if (!scheduler) {
        fprintf(stderr, "Error: Failed to allocate memory for StepLR scheduler\n");
        return NULL;
    }

    scheduler->optimizer = optimizer;
    scheduler->step_size = step_size;
    scheduler->gamma = gamma;
    scheduler->last_epoch = 0;
    scheduler->initial_lr = adam_get_learning_rate(optimizer);

    return scheduler;
}

void step_lr_free(StepLR* scheduler) {
    if (scheduler) {
        free(scheduler);
    }
}

void step_lr_step(StepLR* scheduler) {
    if (!scheduler) {
        fprintf(stderr, "Error: Scheduler is NULL in step_lr_step\n");
        return;
    }

    scheduler->last_epoch++;

    if (scheduler->last_epoch % scheduler->step_size == 0) {
        // Calculate new learning rate: lr = initial_lr * (gamma ^ num_steps)
        float current_lr = adam_get_learning_rate(scheduler->optimizer);
        float new_lr = current_lr * scheduler->gamma;

        adam_set_learning_rate(scheduler->optimizer, new_lr);

        printf("StepLR: Learning rate updated to %.6f at epoch %d\n",
               new_lr, scheduler->last_epoch + 1);
    }
}

float step_lr_get_lr(StepLR* scheduler) {
    if (!scheduler) {
        fprintf(stderr, "Error: Scheduler is NULL in step_lr_get_lr\n");
        return 0.0f;
    }

    return adam_get_learning_rate(scheduler->optimizer);
}

int step_lr_get_last_epoch(StepLR* scheduler) {
    if (!scheduler) {
        fprintf(stderr, "Error: Scheduler is NULL in step_lr_get_last_epoch\n");
        return -1;
    }

    return scheduler->last_epoch;
}

void step_lr_set_last_epoch(StepLR* scheduler, int epoch) {
    if (!scheduler) {
        fprintf(stderr, "Error: Scheduler is NULL in step_lr_set_last_epoch\n");
        return;
    }

    scheduler->last_epoch = epoch;

    int num_steps = epoch / scheduler->step_size;
    float new_lr = scheduler->initial_lr;

    for (int i = 0; i < num_steps; i++) {
        new_lr *= scheduler->gamma;
    }

    adam_set_learning_rate(scheduler->optimizer, new_lr);
}
