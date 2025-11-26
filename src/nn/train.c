#include "nn/cnn.h"
#include "nn/dataset/dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <stdint.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>
#include "image/image.h"

double get_wall_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

void print_progress_bar(int current, int total, const char* prefix, float loss, float acc, float time_per_batch, float total_elapsed_time) {
    const int bar_width = 30;
    float progress = (float)current / total;
    int filled_width = (int)(progress * bar_width);

    char bar[bar_width + 3];
    memset(bar, 0, sizeof(bar));
    bar[0] = '[';
    memset(bar + 1, '#', filled_width);
    memset(bar + 1 + filled_width, '.', bar_width - filled_width);
    bar[bar_width + 1] = ']';
    bar[bar_width + 2] = '\0';

    char eta_str[32] = "";
    if (current > 0 && current < total) {
        float avg_time_per_batch = total_elapsed_time / current;
        float remaining_batches = total - current;
        float eta_seconds = remaining_batches * avg_time_per_batch;

        if (eta_seconds < 60) {
            sprintf(eta_str, " - ETA: %.0fs", eta_seconds);
        } else if (eta_seconds < 3600) {
            int minutes = (int)(eta_seconds / 60);
            int seconds = (int)(eta_seconds) % 60;
            sprintf(eta_str, " - ETA: %dm%ds", minutes, seconds);
        } else {
            int hours = (int)(eta_seconds / 3600);
            int minutes = (int)((eta_seconds - hours * 3600) / 60);
            sprintf(eta_str, " - ETA: %dh%dm", hours, minutes);
        }
    }

    printf("\r%s %s %.1f%% (%d/%d) - Loss: %.4f - Acc: %.2f%% - %.2fms/batch%s",
           prefix, bar, progress * 100, current, total, loss, acc, time_per_batch, eta_str);
    fflush(stdout);

    if (current >= total) {
        printf("\n");
    }
}

float calculate_accuracy(Tensor* predictions, Tensor* targets) {
    if (predictions->shape[0] != targets->shape[0]) {
        fprintf(stderr, "Predictions and targets batch sizes don't match\n");
        return 0.0f;
    }

    int correct = 0;
    int batch_size = predictions->shape[0];
    for (int i = 0; i < batch_size; i++) {
        if ((int)predictions->data[i] == (int)targets->data[i]) {
            correct++;
        }
    }

    return (float)correct / batch_size;
}

void save_model_weights(CNN* model, int epoch) {
    mkdir("weights", 0755);

    #define PATH_BUFFER_SIZE 512
    char conv1_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv1_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn1_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn1_3x3_beta_path[PATH_BUFFER_SIZE];
    char bn1_3x3_running_mean_path[PATH_BUFFER_SIZE];
    char bn1_3x3_running_var_path[PATH_BUFFER_SIZE];
    char conv1_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv1_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn1_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn1_1x1_beta_path[PATH_BUFFER_SIZE];
    char bn1_1x1_running_mean_path[PATH_BUFFER_SIZE];
    char bn1_1x1_running_var_path[PATH_BUFFER_SIZE];
    char shortcut1_weight_path[PATH_BUFFER_SIZE];
    char shortcut1_bias_path[PATH_BUFFER_SIZE];
    char bn_shortcut1_gamma_path[PATH_BUFFER_SIZE];
    char bn_shortcut1_beta_path[PATH_BUFFER_SIZE];
    char bn_shortcut1_running_mean_path[PATH_BUFFER_SIZE];
    char bn_shortcut1_running_var_path[PATH_BUFFER_SIZE];

    char conv2_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv2_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn2_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn2_3x3_beta_path[PATH_BUFFER_SIZE];
    char bn2_3x3_running_mean_path[PATH_BUFFER_SIZE];
    char bn2_3x3_running_var_path[PATH_BUFFER_SIZE];
    char conv2_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv2_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn2_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn2_1x1_beta_path[PATH_BUFFER_SIZE];
    char bn2_1x1_running_mean_path[PATH_BUFFER_SIZE];
    char bn2_1x1_running_var_path[PATH_BUFFER_SIZE];
    char shortcut2_weight_path[PATH_BUFFER_SIZE];
    char shortcut2_bias_path[PATH_BUFFER_SIZE];
    char bn_shortcut2_gamma_path[PATH_BUFFER_SIZE];
    char bn_shortcut2_beta_path[PATH_BUFFER_SIZE];
    char bn_shortcut2_running_mean_path[PATH_BUFFER_SIZE];
    char bn_shortcut2_running_var_path[PATH_BUFFER_SIZE];

    char conv3_3x3_weight_path[PATH_BUFFER_SIZE];
    char conv3_3x3_bias_path[PATH_BUFFER_SIZE];
    char bn3_3x3_gamma_path[PATH_BUFFER_SIZE];
    char bn3_3x3_beta_path[PATH_BUFFER_SIZE];
    char bn3_3x3_running_mean_path[PATH_BUFFER_SIZE];
    char bn3_3x3_running_var_path[PATH_BUFFER_SIZE];
    char conv3_1x1_weight_path[PATH_BUFFER_SIZE];
    char conv3_1x1_bias_path[PATH_BUFFER_SIZE];
    char bn3_1x1_gamma_path[PATH_BUFFER_SIZE];
    char bn3_1x1_beta_path[PATH_BUFFER_SIZE];
    char bn3_1x1_running_mean_path[PATH_BUFFER_SIZE];
    char bn3_1x1_running_var_path[PATH_BUFFER_SIZE];
    char shortcut3_weight_path[PATH_BUFFER_SIZE];
    char shortcut3_bias_path[PATH_BUFFER_SIZE];
    char bn_shortcut3_gamma_path[PATH_BUFFER_SIZE];
    char bn_shortcut3_beta_path[PATH_BUFFER_SIZE];
    char bn_shortcut3_running_mean_path[PATH_BUFFER_SIZE];
    char bn_shortcut3_running_var_path[PATH_BUFFER_SIZE];

    char fc1_weight_path[PATH_BUFFER_SIZE];
    char fc1_bias_path[PATH_BUFFER_SIZE];
    char fc2_weight_path[PATH_BUFFER_SIZE];
    char fc2_bias_path[PATH_BUFFER_SIZE];

    sprintf(conv1_3x3_weight_path, "weights/conv1_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv1_3x3_bias_path, "weights/conv1_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_gamma_path, "weights/bn1_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_beta_path, "weights/bn1_3x3_beta_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_running_mean_path, "weights/bn1_3x3_running_mean_epoch_%d.bin", epoch);
    sprintf(bn1_3x3_running_var_path, "weights/bn1_3x3_running_var_epoch_%d.bin", epoch);
    sprintf(conv1_1x1_weight_path, "weights/conv1_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv1_1x1_bias_path, "weights/conv1_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_gamma_path, "weights/bn1_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_beta_path, "weights/bn1_1x1_beta_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_running_mean_path, "weights/bn1_1x1_running_mean_epoch_%d.bin", epoch);
    sprintf(bn1_1x1_running_var_path, "weights/bn1_1x1_running_var_epoch_%d.bin", epoch);
    sprintf(shortcut1_weight_path, "weights/shortcut1_weight_epoch_%d.bin", epoch);
    sprintf(shortcut1_bias_path, "weights/shortcut1_bias_epoch_%d.bin", epoch);
    sprintf(bn_shortcut1_gamma_path, "weights/bn_shortcut1_gamma_epoch_%d.bin", epoch);
    sprintf(bn_shortcut1_beta_path, "weights/bn_shortcut1_beta_epoch_%d.bin", epoch);
    sprintf(bn_shortcut1_running_mean_path, "weights/bn_shortcut1_running_mean_epoch_%d.bin", epoch);
    sprintf(bn_shortcut1_running_var_path, "weights/bn_shortcut1_running_var_epoch_%d.bin", epoch);

    sprintf(conv2_3x3_weight_path, "weights/conv2_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv2_3x3_bias_path, "weights/conv2_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_gamma_path, "weights/bn2_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_beta_path, "weights/bn2_3x3_beta_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_running_mean_path, "weights/bn2_3x3_running_mean_epoch_%d.bin", epoch);
    sprintf(bn2_3x3_running_var_path, "weights/bn2_3x3_running_var_epoch_%d.bin", epoch);
    sprintf(conv2_1x1_weight_path, "weights/conv2_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv2_1x1_bias_path, "weights/conv2_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_gamma_path, "weights/bn2_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_beta_path, "weights/bn2_1x1_beta_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_running_mean_path, "weights/bn2_1x1_running_mean_epoch_%d.bin", epoch);
    sprintf(bn2_1x1_running_var_path, "weights/bn2_1x1_running_var_epoch_%d.bin", epoch);
    sprintf(shortcut2_weight_path, "weights/shortcut2_weight_epoch_%d.bin", epoch);
    sprintf(shortcut2_bias_path, "weights/shortcut2_bias_epoch_%d.bin", epoch);
    sprintf(bn_shortcut2_gamma_path, "weights/bn_shortcut2_gamma_epoch_%d.bin", epoch);
    sprintf(bn_shortcut2_beta_path, "weights/bn_shortcut2_beta_epoch_%d.bin", epoch);
    sprintf(bn_shortcut2_running_mean_path, "weights/bn_shortcut2_running_mean_epoch_%d.bin", epoch);
    sprintf(bn_shortcut2_running_var_path, "weights/bn_shortcut2_running_var_epoch_%d.bin", epoch);

    sprintf(conv3_3x3_weight_path, "weights/conv3_3x3_weight_epoch_%d.bin", epoch);
    sprintf(conv3_3x3_bias_path, "weights/conv3_3x3_bias_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_gamma_path, "weights/bn3_3x3_gamma_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_beta_path, "weights/bn3_3x3_beta_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_running_mean_path, "weights/bn3_3x3_running_mean_epoch_%d.bin", epoch);
    sprintf(bn3_3x3_running_var_path, "weights/bn3_3x3_running_var_epoch_%d.bin", epoch);
    sprintf(conv3_1x1_weight_path, "weights/conv3_1x1_weight_epoch_%d.bin", epoch);
    sprintf(conv3_1x1_bias_path, "weights/conv3_1x1_bias_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_gamma_path, "weights/bn3_1x1_gamma_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_beta_path, "weights/bn3_1x1_beta_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_running_mean_path, "weights/bn3_1x1_running_mean_epoch_%d.bin", epoch);
    sprintf(bn3_1x1_running_var_path, "weights/bn3_1x1_running_var_epoch_%d.bin", epoch);
    sprintf(shortcut3_weight_path, "weights/shortcut3_weight_epoch_%d.bin", epoch);
    sprintf(shortcut3_bias_path, "weights/shortcut3_bias_epoch_%d.bin", epoch);
    sprintf(bn_shortcut3_gamma_path, "weights/bn_shortcut3_gamma_epoch_%d.bin", epoch);
    sprintf(bn_shortcut3_beta_path, "weights/bn_shortcut3_beta_epoch_%d.bin", epoch);
    sprintf(bn_shortcut3_running_mean_path, "weights/bn_shortcut3_running_mean_epoch_%d.bin", epoch);
    sprintf(bn_shortcut3_running_var_path, "weights/bn_shortcut3_running_var_epoch_%d.bin", epoch);

    sprintf(fc1_weight_path, "weights/fc1_weight_epoch_%d.bin", epoch);
    sprintf(fc1_bias_path, "weights/fc1_bias_epoch_%d.bin", epoch);
    sprintf(fc2_weight_path, "weights/fc2_weight_epoch_%d.bin", epoch);
    sprintf(fc2_bias_path, "weights/fc2_bias_epoch_%d.bin", epoch);

    FILE* f;
    if ((f = fopen(conv1_3x3_weight_path, "wb"))) {
        fwrite(model->conv1_3x3->weight->data, sizeof(float), model->conv1_3x3->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv1_3x3_bias_path, "wb"))) {
        fwrite(model->conv1_3x3->bias->data, sizeof(float), model->conv1_3x3->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_3x3_gamma_path, "wb"))) {
        fwrite(model->bn1_3x3->layer_grad->weights->data, sizeof(float), model->bn1_3x3->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_3x3_beta_path, "wb"))) {
        fwrite(model->bn1_3x3->layer_grad->biases->data, sizeof(float), model->bn1_3x3->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_3x3_running_mean_path, "wb"))) {
        fwrite(model->bn1_3x3->running_mean->data, sizeof(float), model->bn1_3x3->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_3x3_running_var_path, "wb"))) {
        fwrite(model->bn1_3x3->running_var->data, sizeof(float), model->bn1_3x3->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(conv1_1x1_weight_path, "wb"))) {
        fwrite(model->conv1_1x1->weight->data, sizeof(float), model->conv1_1x1->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv1_1x1_bias_path, "wb"))) {
        fwrite(model->conv1_1x1->bias->data, sizeof(float), model->conv1_1x1->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_1x1_gamma_path, "wb"))) {
        fwrite(model->bn1_1x1->layer_grad->weights->data, sizeof(float), model->bn1_1x1->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_1x1_beta_path, "wb"))) {
        fwrite(model->bn1_1x1->layer_grad->biases->data, sizeof(float), model->bn1_1x1->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_1x1_running_mean_path, "wb"))) {
        fwrite(model->bn1_1x1->running_mean->data, sizeof(float), model->bn1_1x1->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn1_1x1_running_var_path, "wb"))) {
        fwrite(model->bn1_1x1->running_var->data, sizeof(float), model->bn1_1x1->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut1_weight_path, "wb"))) {
        fwrite(model->shortcut1->weight->data, sizeof(float), model->shortcut1->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut1_bias_path, "wb"))) {
        fwrite(model->shortcut1->bias->data, sizeof(float), model->shortcut1->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut1_gamma_path, "wb"))) {
        fwrite(model->bn_shortcut1->layer_grad->weights->data, sizeof(float), model->bn_shortcut1->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut1_beta_path, "wb"))) {
        fwrite(model->bn_shortcut1->layer_grad->biases->data, sizeof(float), model->bn_shortcut1->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut1_running_mean_path, "wb"))) {
        fwrite(model->bn_shortcut1->running_mean->data, sizeof(float), model->bn_shortcut1->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut1_running_var_path, "wb"))) {
        fwrite(model->bn_shortcut1->running_var->data, sizeof(float), model->bn_shortcut1->running_var->size, f);
        fclose(f);
    }

    if ((f = fopen(conv2_3x3_weight_path, "wb"))) {
        fwrite(model->conv2_3x3->weight->data, sizeof(float), model->conv2_3x3->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv2_3x3_bias_path, "wb"))) {
        fwrite(model->conv2_3x3->bias->data, sizeof(float), model->conv2_3x3->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_3x3_gamma_path, "wb"))) {
        fwrite(model->bn2_3x3->layer_grad->weights->data, sizeof(float), model->bn2_3x3->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_3x3_beta_path, "wb"))) {
        fwrite(model->bn2_3x3->layer_grad->biases->data, sizeof(float), model->bn2_3x3->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_3x3_running_mean_path, "wb"))) {
        fwrite(model->bn2_3x3->running_mean->data, sizeof(float), model->bn2_3x3->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_3x3_running_var_path, "wb"))) {
        fwrite(model->bn2_3x3->running_var->data, sizeof(float), model->bn2_3x3->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(conv2_1x1_weight_path, "wb"))) {
        fwrite(model->conv2_1x1->weight->data, sizeof(float), model->conv2_1x1->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv2_1x1_bias_path, "wb"))) {
        fwrite(model->conv2_1x1->bias->data, sizeof(float), model->conv2_1x1->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_1x1_gamma_path, "wb"))) {
        fwrite(model->bn2_1x1->layer_grad->weights->data, sizeof(float), model->bn2_1x1->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_1x1_beta_path, "wb"))) {
        fwrite(model->bn2_1x1->layer_grad->biases->data, sizeof(float), model->bn2_1x1->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_1x1_running_mean_path, "wb"))) {
        fwrite(model->bn2_1x1->running_mean->data, sizeof(float), model->bn2_1x1->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn2_1x1_running_var_path, "wb"))) {
        fwrite(model->bn2_1x1->running_var->data, sizeof(float), model->bn2_1x1->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut2_weight_path, "wb"))) {
        fwrite(model->shortcut2->weight->data, sizeof(float), model->shortcut2->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut2_bias_path, "wb"))) {
        fwrite(model->shortcut2->bias->data, sizeof(float), model->shortcut2->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut2_gamma_path, "wb"))) {
        fwrite(model->bn_shortcut2->layer_grad->weights->data, sizeof(float), model->bn_shortcut2->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut2_beta_path, "wb"))) {
        fwrite(model->bn_shortcut2->layer_grad->biases->data, sizeof(float), model->bn_shortcut2->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut2_running_mean_path, "wb"))) {
        fwrite(model->bn_shortcut2->running_mean->data, sizeof(float), model->bn_shortcut2->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut2_running_var_path, "wb"))) {
        fwrite(model->bn_shortcut2->running_var->data, sizeof(float), model->bn_shortcut2->running_var->size, f);
        fclose(f);
    }

    if ((f = fopen(conv3_3x3_weight_path, "wb"))) {
        fwrite(model->conv3_3x3->weight->data, sizeof(float), model->conv3_3x3->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv3_3x3_bias_path, "wb"))) {
        fwrite(model->conv3_3x3->bias->data, sizeof(float), model->conv3_3x3->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_3x3_gamma_path, "wb"))) {
        fwrite(model->bn3_3x3->layer_grad->weights->data, sizeof(float), model->bn3_3x3->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_3x3_beta_path, "wb"))) {
        fwrite(model->bn3_3x3->layer_grad->biases->data, sizeof(float), model->bn3_3x3->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_3x3_running_mean_path, "wb"))) {
        fwrite(model->bn3_3x3->running_mean->data, sizeof(float), model->bn3_3x3->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_3x3_running_var_path, "wb"))) {
        fwrite(model->bn3_3x3->running_var->data, sizeof(float), model->bn3_3x3->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(conv3_1x1_weight_path, "wb"))) {
        fwrite(model->conv3_1x1->weight->data, sizeof(float), model->conv3_1x1->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(conv3_1x1_bias_path, "wb"))) {
        fwrite(model->conv3_1x1->bias->data, sizeof(float), model->conv3_1x1->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_1x1_gamma_path, "wb"))) {
        fwrite(model->bn3_1x1->layer_grad->weights->data, sizeof(float), model->bn3_1x1->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_1x1_beta_path, "wb"))) {
        fwrite(model->bn3_1x1->layer_grad->biases->data, sizeof(float), model->bn3_1x1->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_1x1_running_mean_path, "wb"))) {
        fwrite(model->bn3_1x1->running_mean->data, sizeof(float), model->bn3_1x1->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn3_1x1_running_var_path, "wb"))) {
        fwrite(model->bn3_1x1->running_var->data, sizeof(float), model->bn3_1x1->running_var->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut3_weight_path, "wb"))) {
        fwrite(model->shortcut3->weight->data, sizeof(float), model->shortcut3->weight->size, f);
        fclose(f);
    }
    if ((f = fopen(shortcut3_bias_path, "wb"))) {
        fwrite(model->shortcut3->bias->data, sizeof(float), model->shortcut3->bias->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut3_gamma_path, "wb"))) {
        fwrite(model->bn_shortcut3->layer_grad->weights->data, sizeof(float), model->bn_shortcut3->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut3_beta_path, "wb"))) {
        fwrite(model->bn_shortcut3->layer_grad->biases->data, sizeof(float), model->bn_shortcut3->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut3_running_mean_path, "wb"))) {
        fwrite(model->bn_shortcut3->running_mean->data, sizeof(float), model->bn_shortcut3->running_mean->size, f);
        fclose(f);
    }
    if ((f = fopen(bn_shortcut3_running_var_path, "wb"))) {
        fwrite(model->bn_shortcut3->running_var->data, sizeof(float), model->bn_shortcut3->running_var->size, f);
        fclose(f);
    }

    if ((f = fopen(fc1_weight_path, "wb"))) {
        fwrite(model->fc1->layer_grad->weights->data, sizeof(float), model->fc1->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(fc1_bias_path, "wb"))) {
        fwrite(model->fc1->layer_grad->biases->data, sizeof(float), model->fc1->layer_grad->biases->size, f);
        fclose(f);
    }
    if ((f = fopen(fc2_weight_path, "wb"))) {
        fwrite(model->fc2->layer_grad->weights->data, sizeof(float), model->fc2->layer_grad->weights->size, f);
        fclose(f);
    }
    if ((f = fopen(fc2_bias_path, "wb"))) {
        fwrite(model->fc2->layer_grad->biases->data, sizeof(float), model->fc2->layer_grad->biases->size, f);
        fclose(f);
    }

    printf("  Weights saved for epoch %d\n", epoch);
    #undef PATH_BUFFER_SIZE
}

void save_loss_data(int epoch, float train_loss, float train_acc, float test_acc) {
    mkdir("training_data/logs", 0755);
    FILE* f = fopen("training_data/logs/training_log.txt", "a");
    if (f) {
        fprintf(f, "%d %.6f %.4f %.4f\n", epoch, train_loss, train_acc, test_acc);
        fclose(f);
    }
}

void save_additional_metrics(int epoch, float learning_rate, float grad_norm, float lr_decay_factor, int patience_counter) {
    mkdir("training_data/metrics", 0755);
    FILE* f = fopen("training_data/metrics/additional_metrics.txt", "a");
    if (f) {
        fprintf(f, "%d %.6f %.6f %.6f %d\n", epoch, learning_rate, grad_norm, lr_decay_factor, patience_counter);
        fclose(f);
    }
}

// Calculate gradient norm for monitoring training stability
float calculate_gradient_norm(CNN* model) {
    float total_norm = 0.0f;
    int param_count = 0;

    // Calculate norm for conv1_3x3 gradients
    if (model->conv1_3x3->weight_grad) {
        for (int i = 0; i < model->conv1_3x3->weight_grad->size; i++) {
            total_norm += model->conv1_3x3->weight_grad->data[i] * model->conv1_3x3->weight_grad->data[i];
        }
        param_count += model->conv1_3x3->weight_grad->size;
    }

    if (model->conv1_3x3->bias_grad) {
        for (int i = 0; i < model->conv1_3x3->bias_grad->size; i++) {
            total_norm += model->conv1_3x3->bias_grad->data[i] * model->conv1_3x3->bias_grad->data[i];
        }
        param_count += model->conv1_3x3->bias_grad->size;
    }

    // Add other layer gradients...
    // For brevity, just calculating conv1_3x3 for now - can expand later

    return sqrtf(total_norm / param_count);  // RMS gradient norm
}

void save_training_metadata(int num_epochs, int batch_size, float initial_lr, float weight_decay, int patience, char* dataset_info) {
    mkdir("training_data/metadata", 0755);

    FILE* f = fopen("training_data/metadata/training_config.txt", "w");
    if (f) {
        time_t now = time(NULL);
        fprintf(f, "Training Configuration\n");
        fprintf(f, "=====================\n");
        fprintf(f, "Timestamp: %s", ctime(&now));
        fprintf(f, "Dataset: %s\n", dataset_info);
        fprintf(f, "Epochs: %d\n", num_epochs);
        fprintf(f, "Batch Size: %d\n", batch_size);
        fprintf(f, "Initial Learning Rate: %.6f\n", initial_lr);
        fprintf(f, "Weight Decay: %.6f\n", weight_decay);
        fprintf(f, "Early Stopping Patience: %d\n", patience);
        fprintf(f, "Optimizer: Adam (beta1=0.9, beta2=0.999, epsilon=1e-8)\n");
        fprintf(f, "Scheduler: StepLR (step_size=7, gamma=0.1)\n");
        fprintf(f, "Input Normalization: [-1, 1] (from [0,1])\n");
        fprintf(f, "\nData Format Descriptions:\n");
        fprintf(f, "- training_log.txt: epoch, train_loss, train_accuracy, test_accuracy\n");
        fprintf(f, "- batch_log.txt: epoch, batch_idx, batch_loss, batch_accuracy, time_ms\n");
        fprintf(f, "- additional_metrics.txt: epoch, learning_rate, gradient_norm, lr_decay_factor, patience_counter\n");
        fprintf(f, "- batch_log_epoch_X.txt: per-epoch batch data\n");
        fprintf(f, "- weights/: binary weight files per epoch\n");
        fclose(f);
        printf("Training metadata saved to training_data/metadata/training_config.txt\n");
    }
}


void save_batch_data(int epoch, int batch_idx, float batch_loss, float batch_acc, float time_ms) {
    mkdir("training_data/logs", 0755);
    FILE* f = fopen("training_data/logs/batch_log.txt", "a");
    if (f) {
        fprintf(f, "%d,%d,%.6f,%.4f,%.2f\n", epoch, batch_idx, batch_loss, batch_acc, time_ms);
        fclose(f);
    }
}



void save_epoch_batch_data(int epoch, float* batch_losses, float* batch_accuracies, float* batch_times, int num_batches) {
    mkdir("training_data/batch_data", 0755);
    char filename[256];
    sprintf(filename, "training_data/batch_data/batch_log_epoch_%d.txt", epoch);

    FILE* f = fopen(filename, "w");
    if (f) {
        fprintf(f, "batch_idx,loss,accuracy,time_ms\n");

        for (int i = 0; i < num_batches; i++) {
            fprintf(f, "%d,%.6f,%.4f,%.2f\n", i + 1, batch_losses[i], batch_accuracies[i], batch_times[i]);
        }

        fclose(f);
        printf("  Batch data saved to %s\n", filename);
    } else {
        fprintf(stderr, "Failed to create batch log file for epoch %d\n", epoch);
    }
}

// Setup optimizer and scheduler for the model
void setup_optimizer_scheduler(CNN* model) {
    Tensor** params;
    Tensor** grads;
    int num_params = cnn_get_parameters(model, &params, &grads);

    model->optimizer = adam_create(1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);
    for (int i = 0; i < num_params; i++) {
        adam_add_param(model->optimizer, params[i], grads[i]);
    }

    model->scheduler = step_lr_create(model->optimizer, 7, 0.1f);

    cnn_free_parameters(params, grads);
}

int main(int argc, char* argv[]) {
    printf("EMNIST Lowercase Letter CNN Training\n");
    printf("=====================================\n\n");

    omp_set_num_threads(omp_get_num_procs());
    omp_set_nested(0);

    printf("OpenMP configured to use %d threads\n", omp_get_max_threads());

    int custom_epochs = -1;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            custom_epochs = atoi(argv[i + 1]);
            i++;
            printf("Custom epochs: %d\n", custom_epochs);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--epochs N]\n", argv[0]);
            return 1;
        }
    }

    srand(time(NULL));

    printf("Loading EMNIST Lowercase Letters dataset...\n");
    Dataset* train_dataset = dataset_load_emnist("data/font_letters_train-images.idx",
                                                "data/font_letters_train-labels.idx",
                                                64, 1, 4, NULL);

    Dataset* test_dataset = dataset_load_emnist("data/font_letters_test-images.idx",
                                               "data/font_letters_test-labels.idx",
                                               1000, 0, 1, NULL);
    if (!train_dataset || !test_dataset) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }

    if (!train_dataset->batches || !test_dataset->batches) {
        fprintf(stderr, "Dataset batches not allocated\n");
        return 1;
    }

    printf("Creating CNN model...\n");
    CNN* model = cnn_create();
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    setup_optimizer_scheduler(model);

    int num_epochs = (custom_epochs > 0) ? custom_epochs : 12;
    float best_accuracy = 0.0f;
    int patience = 10;
    int patience_counter = 0;

    char dataset_info[256];
    sprintf(dataset_info, "EMNIST Lowercase Letters (filtered from byclass, %d train samples, %d test samples)",
            train_dataset->total_samples, test_dataset->total_samples);
    save_training_metadata(num_epochs, 64, 1e-3f, 1e-4f, patience, dataset_info);

    printf("Starting training for %d epochs...\n\n", num_epochs);

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Adam Optimizer Parameters:\n");
        printf("  Learning Rate: %.6f\n", adam_get_learning_rate(model->optimizer));
        printf("  Beta1: %.6f\n", adam_get_beta1(model->optimizer));
        printf("  Beta2: %.6f\n", adam_get_beta2(model->optimizer));
        printf("  Epsilon: %.2e\n", adam_get_epsilon(model->optimizer));
        printf("  Weight Decay: %.2e\n", adam_get_weight_decay(model->optimizer));
        printf("  Time Step (t): %d\n", adam_get_t(model->optimizer));
        printf("  Number of Parameters: %d\n", adam_get_num_params(model->optimizer));
        printf("\n");

        printf("Epoch %d/%d\n", epoch + 1, num_epochs);
        printf("----------\n");

        cnn_train(model);
        float epoch_loss = 0.0f;
        int total_train_samples = 0;
        int correct_train = 0;

        float* batch_losses = (float*)malloc(train_dataset->num_batches * sizeof(float));
        float* batch_accuracies = (float*)malloc(train_dataset->num_batches * sizeof(float));
        float* batch_times = (float*)malloc(train_dataset->num_batches * sizeof(float));

        float total_elapsed_time = 0.0f;

        for (int batch_idx = 0; batch_idx < train_dataset->num_batches; batch_idx++) {
            double batch_start = get_wall_time_ms();

            Batch* batch = &train_dataset->batches[batch_idx];

            // Normalize input from [0,1] to [-1,1]
            Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < batch->data->size; i++) {
                input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;
            }

            // Labels are in range 0-25
            int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
            Tensor* targets_adjusted = tensor_create(targets_shape, 4);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < batch->labels->size; i++) {
                targets_adjusted->data[i] = batch->labels->data[i];
            }

            CNNForwardResult* forward_result = cnn_forward(model, input_normalized);
            if (!forward_result) {
                fprintf(stderr, "Forward pass failed\n");
                tensor_free(input_normalized);
                tensor_free(targets_adjusted);
                free(batch_losses);
                free(batch_accuracies);
                free(batch_times);
                return 1;
            }

            // Training step (loss + backward + optimizer)
            float batch_loss = cnn_training_step(model, forward_result, targets_adjusted);
            epoch_loss += batch_loss * batch->data->shape[0];  // batch_loss is averaged, so multiply back by batch_size
            // Calculate training accuracy for this batch (using same forward pass as loss)
            int pred_shape[] = {batch->data->shape[0], 1, 1, 1};
            Tensor* predictions = tensor_create(pred_shape, 4);
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < batch->data->shape[0]; b++) {
                int max_idx = 0;
                float max_val = forward_result->fc2_out->data[b * 26];
                for (int c = 1; c < 26; c++) {
                    float val = forward_result->fc2_out->data[b * 26 + c];
                    if (val > max_val) {
                        max_val = val;
                        max_idx = c;
                    }
                }
                predictions->data[b] = (float)max_idx;
            }
            float batch_acc = calculate_accuracy(predictions, targets_adjusted) * 100.0f;
            correct_train += (int)(batch_acc * batch->data->shape[0] / 100.0f);
            total_train_samples += batch->data->shape[0];
            double batch_end = get_wall_time_ms();
            float time_per_batch = (float)(batch_end - batch_start);

            total_elapsed_time += time_per_batch / 1000.0f;

            batch_losses[batch_idx] = batch_loss;
            batch_accuracies[batch_idx] = batch_acc;
            batch_times[batch_idx] = time_per_batch;
            tensor_free(predictions);
            cnn_forward_result_free(forward_result);
            tensor_free(input_normalized);
            tensor_free(targets_adjusted);
            float current_avg_loss = epoch_loss / total_train_samples;
            float current_avg_acc = (float)correct_train / total_train_samples * 100.0f;
            char prefix[32];
            print_progress_bar(batch_idx + 1, train_dataset->num_batches, prefix, current_avg_loss, current_avg_acc, time_per_batch, total_elapsed_time);
        }

        epoch_loss /= total_train_samples;
        float epoch_train_acc = (float)correct_train / total_train_samples * 100.0f;

        printf("\n  Train Loss: %.4f, Train Acc: %.2f%%\n", epoch_loss, epoch_train_acc);

        save_epoch_batch_data(epoch + 1, batch_losses, batch_accuracies, batch_times, train_dataset->num_batches);

        free(batch_losses);
        free(batch_accuracies);
        free(batch_times);

        cnn_step_scheduler(model);

        cnn_eval(model);
        int correct_test = 0;
        int total_test_samples = 0;

        for (int batch_idx = 0; batch_idx < test_dataset->num_batches; batch_idx++) {
            Batch* batch = &test_dataset->batches[batch_idx];

            Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < batch->data->size; i++) {
                input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;
            }

            int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
            Tensor* targets_adjusted = tensor_create(targets_shape, 4);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < batch->labels->size; i++) {
                targets_adjusted->data[i] = batch->labels->data[i];
            }

            Tensor* predictions = cnn_predict(model, input_normalized);
            float batch_acc = calculate_accuracy(predictions, targets_adjusted);
            correct_test += (int)(batch_acc * batch->data->shape[0]);
            total_test_samples += batch->data->shape[0];

            tensor_free(predictions);
            tensor_free(input_normalized);
            tensor_free(targets_adjusted);

            float current_test_acc = (float)correct_test / total_test_samples * 100.0f;
            print_progress_bar(batch_idx + 1, test_dataset->num_batches, "Testing", 0.0f, current_test_acc, 0.0f, 0.0f);
        }

        float test_accuracy = (float)correct_test / total_test_samples * 100.0f;
        printf("  Test Accuracy: %.2f%%\n\n", test_accuracy);

        float grad_norm = calculate_gradient_norm(model);
        float current_lr = adam_get_learning_rate(model->optimizer);
        float lr_decay_factor = 0.1f;
        save_additional_metrics(epoch + 1, current_lr, grad_norm, lr_decay_factor, patience_counter);

        save_model_weights(model, epoch + 1);
        save_loss_data(epoch + 1, epoch_loss, epoch_train_acc, test_accuracy);

        if (test_accuracy > best_accuracy) {
            best_accuracy = test_accuracy;
            patience_counter = 0;
            printf("  New best accuracy: %.2f%%\n", best_accuracy);
        } else {
            patience_counter++;
            printf("  No improvement for %d epochs\n", patience_counter);
        }

        if (patience_counter >= patience) {
            printf("Early stopping triggered after %d epochs\n", epoch + 1);
            break;
        }
    }

    printf("\nTraining completed. Best test accuracy: %.2f%%\n", best_accuracy);

    cnn_eval(model);
    int final_correct = 0;
    int final_total = 0;

    for (int batch_idx = 0; batch_idx < test_dataset->num_batches; batch_idx++) {
        Batch* batch = &test_dataset->batches[batch_idx];

        Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batch->data->size; i++) {
            input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;
        }

        int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
        Tensor* targets_adjusted = tensor_create(targets_shape, 4);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < batch->labels->size; i++) {
            targets_adjusted->data[i] = batch->labels->data[i];
        }

        Tensor* predictions = cnn_predict(model, input_normalized);
        float batch_acc = calculate_accuracy(predictions, targets_adjusted);
        final_correct += (int)(batch_acc * batch->data->shape[0]);
        final_total += batch->data->shape[0];

        tensor_free(predictions);
        tensor_free(input_normalized);
        tensor_free(targets_adjusted);

        float current_final_acc = (float)final_correct / final_total * 100.0f;
        print_progress_bar(batch_idx + 1, test_dataset->num_batches, "Final Testing", 0.0f, current_final_acc, 0.0f, 0.0f);
    }

    float final_accuracy = (float)final_correct / final_total * 100.0f;
    printf("Final Test Accuracy: %.2f%%\n", final_accuracy);

    dataset_free(train_dataset);
    dataset_free(test_dataset);
    cnn_free(model);

    return 0;
}
