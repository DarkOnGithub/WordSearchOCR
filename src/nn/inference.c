#include "../include/nn/inference.h"

CNN* cnn_load_model(int epoch) {
    CNN* model = cnn_create();
    if (!model) {
        fprintf(stderr, "Failed to create CNN model\n");
        return NULL;
    }

    if (!cnn_load_weights(model, epoch)) {
        fprintf(stderr, "Failed to load weights from epoch %d\n", epoch);
        cnn_free(model);
        return NULL;
    }

    cnn_eval(model);

    return model;
}

void cnn_free_model(CNN* model) {
    if (model) {
        cnn_free(model);
    }
}