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
#include "image/image.h"

// Get current time in milliseconds (wall time)
double get_wall_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// Format time duration into human-readable string
void format_time_duration(double seconds, char* buffer, size_t buffer_size) {
    if (seconds < 60) {
        snprintf(buffer, buffer_size, "%.0fs", seconds);
    } else if (seconds < 3600) {
        int minutes = (int)(seconds / 60);
        int secs = (int)(seconds) % 60;
        snprintf(buffer, buffer_size, "%dm%ds", minutes, secs);
    } else {
        int hours = (int)(seconds / 3600);
        int minutes = (int)((seconds - hours * 3600) / 60);
        snprintf(buffer, buffer_size, "%dh%dm", hours, minutes);
    }
}

// Simple progress bar function (tqdm-like)
void print_progress_bar(int current, int total, const char* prefix, float loss, float acc, float time_per_batch, float total_elapsed_time) {
    const int bar_width = 30;
    float progress = (float)current / total;
    int filled_width = (int)(progress * bar_width);

    // Create progress bar string
    char bar[bar_width + 3];  // +3 for [ ]
    memset(bar, 0, sizeof(bar));
    bar[0] = '[';
    memset(bar + 1, '#', filled_width);
    memset(bar + 1 + filled_width, '.', bar_width - filled_width);
    bar[bar_width + 1] = ']';
    bar[bar_width + 2] = '\0';

    // Calculate ETA (Estimated Time of Arrival)
    char eta_str[32] = "";
    if (current > 0 && current < total) {
        float avg_time_per_batch = total_elapsed_time / current;
        float remaining_batches = total - current;
        float eta_seconds = remaining_batches * avg_time_per_batch;

        // Format ETA
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

    // Print progress bar with metrics and ETA
    printf("\r%s %s %.1f%% (%d/%d) - Loss: %.4f - Acc: %.2f%% - %.2fms/batch%s",
           prefix, bar, progress * 100, current, total, loss, acc, time_per_batch, eta_str);
    fflush(stdout);

    // New line when complete
    if (current >= total) {
        printf("\n");
    }
}

// Calculate accuracy from predictions and targets
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

// Save model weights to binary files
void save_model_weights(CNN* model, int epoch) {
    // Create weights directory if it doesn't exist
    mkdir("training_data/weights", 0755);

    char filename[256];

    // Save conv1_3x3 weights and bias
    sprintf(filename, "training_data/weights/conv1_3x3_weight_epoch_%d.bin", epoch);
    FILE* f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv1_3x3->weight->data, sizeof(float), model->conv1_3x3->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv1_3x3_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv1_3x3->bias->data, sizeof(float), model->conv1_3x3->bias->size, f);
        fclose(f);
    }

    // Save conv1_1x1 weights and bias
    sprintf(filename, "training_data/weights/conv1_1x1_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv1_1x1->weight->data, sizeof(float), model->conv1_1x1->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv1_1x1_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv1_1x1->bias->data, sizeof(float), model->conv1_1x1->bias->size, f);
        fclose(f);
    }

    // Save conv2_3x3 weights and bias
    sprintf(filename, "training_data/weights/conv2_3x3_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv2_3x3->weight->data, sizeof(float), model->conv2_3x3->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv2_3x3_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv2_3x3->bias->data, sizeof(float), model->conv2_3x3->bias->size, f);
        fclose(f);
    }

    // Save conv2_1x1 weights and bias
    sprintf(filename, "training_data/weights/conv2_1x1_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv2_1x1->weight->data, sizeof(float), model->conv2_1x1->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv2_1x1_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv2_1x1->bias->data, sizeof(float), model->conv2_1x1->bias->size, f);
        fclose(f);
    }

    // Save conv3_3x3 weights and bias
    sprintf(filename, "training_data/weights/conv3_3x3_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv3_3x3->weight->data, sizeof(float), model->conv3_3x3->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv3_3x3_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv3_3x3->bias->data, sizeof(float), model->conv3_3x3->bias->size, f);
        fclose(f);
    }

    // Save conv3_1x1 weights and bias
    sprintf(filename, "training_data/weights/conv3_1x1_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv3_1x1->weight->data, sizeof(float), model->conv3_1x1->weight->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/conv3_1x1_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->conv3_1x1->bias->data, sizeof(float), model->conv3_1x1->bias->size, f);
        fclose(f);
    }

    // Save fc1 weights and bias
    sprintf(filename, "training_data/weights/fc1_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->fc1->layer_grad->weights->data, sizeof(float), model->fc1->layer_grad->weights->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/fc1_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->fc1->layer_grad->biases->data, sizeof(float), model->fc1->layer_grad->biases->size, f);
        fclose(f);
    }

    // Save fc2 weights and bias
    sprintf(filename, "training_data/weights/fc2_weight_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->fc2->layer_grad->weights->data, sizeof(float), model->fc2->layer_grad->weights->size, f);
        fclose(f);
    }

    sprintf(filename, "training_data/weights/fc2_bias_epoch_%d.bin", epoch);
    f = fopen(filename, "wb");
    if (f) {
        fwrite(model->fc2->layer_grad->biases->data, sizeof(float), model->fc2->layer_grad->biases->size, f);
        fclose(f);
    }

    printf("  Weights saved for epoch %d\n", epoch);
}

// Save loss data to text file (compact format - space separated)
void save_loss_data(int epoch, float train_loss, float train_acc, float test_acc) {
    // Create logs directory if it doesn't exist
    mkdir("training_data/logs", 0755);
    FILE* f = fopen("training_data/logs/training_log.txt", "a");
    if (f) {
        fprintf(f, "%d %.6f %.4f %.4f\n", epoch, train_loss, train_acc, test_acc);
        fclose(f);
    }
}

// Save additional metrics (learning rate, gradient norms, etc.)
void save_additional_metrics(int epoch, float learning_rate, float grad_norm, float lr_decay_factor, int patience_counter) {
    // Create metrics directory if it doesn't exist
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

// Save training metadata and configuration
void save_training_metadata(int num_epochs, int batch_size, float initial_lr, float weight_decay, int patience, char* dataset_info) {
    // Create metadata directory if it doesn't exist
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

// Save model checkpoint (full model state)
void save_checkpoint(CNN* model, int epoch, float best_accuracy, float current_accuracy, char* checkpoint_type) {
    // Create checkpoints directory if it doesn't exist
    mkdir("training_data/checkpoints", 0755);

    char filename[256];
    if (strcmp(checkpoint_type, "best") == 0) {
        sprintf(filename, "training_data/checkpoints/best_model_epoch_%d_acc_%.2f.bin", epoch, best_accuracy);
    } else if (strcmp(checkpoint_type, "latest") == 0) {
        sprintf(filename, "training_data/checkpoints/latest_checkpoint.bin");
    } else if (strcmp(checkpoint_type, "regular") == 0) {
        sprintf(filename, "training_data/checkpoints/checkpoint_epoch_%d.bin", epoch);
    }

    // For now, just save weights - can extend to save optimizer state later
    save_model_weights(model, epoch);

    // Save checkpoint metadata
    char meta_filename[256];
    sprintf(meta_filename, "%s.meta", filename);
    FILE* f = fopen(meta_filename, "w");
    if (f) {
        fprintf(f, "epoch=%d\n", epoch);
        fprintf(f, "accuracy=%.4f\n", current_accuracy);
        fprintf(f, "best_accuracy=%.4f\n", best_accuracy);
        fprintf(f, "type=%s\n", checkpoint_type);
        fclose(f);
    }

    printf("  Checkpoint saved: %s\n", checkpoint_type);
}

// Save batch-level data to text file (compact format for graphing: epoch,batch_idx,loss,accuracy,time_ms)
void save_batch_data(int epoch, int batch_idx, float batch_loss, float batch_acc, float time_ms) {
    // Create logs directory if it doesn't exist
    mkdir("training_data/logs", 0755);
    FILE* f = fopen("training_data/logs/batch_log.txt", "a");
    if (f) {
        fprintf(f, "%d,%d,%.6f,%.4f,%.2f\n", epoch, batch_idx, batch_loss, batch_acc, time_ms);
        fclose(f);
    }
}

// Save all batch data for an epoch to a separate file
void save_epoch_batch_data(int epoch, float* batch_losses, float* batch_accuracies, float* batch_times, int num_batches) {
    // Create batch_data directory if it doesn't exist
    mkdir("training_data/batch_data", 0755);
    char filename[256];
    sprintf(filename, "training_data/batch_data/batch_log_epoch_%d.txt", epoch);

    FILE* f = fopen(filename, "w");
    if (f) {
        // Write header
        fprintf(f, "batch_idx,loss,accuracy,time_ms\n");

        // Write all batch data for this epoch
        for (int i = 0; i < num_batches; i++) {
            fprintf(f, "%d,%.6f,%.4f,%.2f\n", i + 1, batch_losses[i], batch_accuracies[i], batch_times[i]);
        }

        fclose(f);
        printf("  Batch data saved to %s\n", filename);
    } else {
        fprintf(stderr, "Failed to create batch log file for epoch %d\n", epoch);
    }
}

// Save a single image as PNG format using Image struct
void save_image_png(const Tensor* image, const char* filename) {
    if (!image || image->ndim != 4 || image->shape[0] != 1 || image->shape[2] != 28 || image->shape[3] != 28) {
        fprintf(stderr, "Error: Invalid image tensor format (expected 1x1x28x28)\n");
        return;
    }

    // Create Image struct for grayscale image
    Image img;
    img.width = 28;
    img.height = 28;
    img.is_grayscale = true;
    img.gray_pixels = (uint8_t*)malloc(28 * 28 * sizeof(uint8_t));
    img.rgba_pixels = NULL;

    if (!img.gray_pixels) {
        fprintf(stderr, "Error: Failed to allocate memory for image pixels\n");
        return;
    }

    // Convert tensor data (float 0-1) to uint8_t (0-255)
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            // Tensor is in NCHW format: [batch, channel, height, width]
            int idx = 0 * 1 * 28 * 28 + 0 * 28 * 28 + y * 28 + x;
            float pixel = image->data[idx];
            // Clamp to [0,1] and convert to [0,255]
            int pixel_int = (int)(pixel * 255.0f + 0.5f);
            if (pixel_int < 0) pixel_int = 0;
            if (pixel_int > 255) pixel_int = 255;
            img.gray_pixels[y * 28 + x] = (uint8_t)pixel_int;
        }
    }

    // Save as PNG
    save_image(filename, &img);

    // Free image memory
    free_image(&img);
}

// Save all images from a dataset to a directory
void save_all_images(const Dataset* dataset, const char* output_dir) {
    if (!dataset) {
        fprintf(stderr, "Error: Dataset is NULL\n");
        return;
    }

    // Create output directory if it doesn't exist
    mkdir(output_dir, 0755);

    // Create subdirectories for each letter class (A-Z)
    for (int class = 0; class < 26; class++) {
        char class_dir[256];
        sprintf(class_dir, "%s/%c", output_dir, 'A' + class);
        mkdir(class_dir, 0755);
    }

    printf("Saving %d images to %s...\n", dataset->total_samples, output_dir);

    int image_count = 0;
    for (int batch_idx = 0; batch_idx < dataset->num_batches; batch_idx++) {
        Batch* batch = &dataset->batches[batch_idx];

        for (int img_idx = 0; img_idx < batch->data->shape[0]; img_idx++) {
            // Get label for this image
            int label = (int)batch->labels->data[img_idx];

            // Create individual image tensor (1x1x28x28)
            int img_shape[] = {1, 1, 28, 28};
            Tensor* single_image = tensor_create(img_shape, 4);

            // Copy image data from batch
            for (int y = 0; y < 28; y++) {
                for (int x = 0; x < 28; x++) {
                    int batch_idx_4d = img_idx * 1 * 28 * 28 + 0 * 28 * 28 + y * 28 + x;
                    int img_idx_4d = 0 * 1 * 28 * 28 + 0 * 28 * 28 + y * 28 + x;
                    single_image->data[img_idx_4d] = batch->data->data[batch_idx_4d];
                }
            }

            // Create filename: A/00001_A.png, B/00002_B.png, etc.
            char letter = 'A' + label;
            char class_dir[256];
            sprintf(class_dir, "%s/%c", output_dir, letter);
            char filename[512];
            sprintf(filename, "%s/%05d_%c.png", class_dir, image_count + 1, letter);

            // Save the image
            save_image_png(single_image, filename);

            tensor_free(single_image);
            image_count++;

            // Progress indicator
            if (image_count % 1000 == 0 || image_count == dataset->total_samples) {
                printf("  Saved %d/%d images\r", image_count, dataset->total_samples);
                fflush(stdout);
            }
        }
    }

    printf("\n✓ Successfully saved %d images to %s\n", image_count, output_dir);
}

// Setup optimizer and scheduler for the model
void setup_optimizer_scheduler(CNN* model) {
    // Get all model parameters
    Tensor** params;
    Tensor** grads;
    int num_params = cnn_get_parameters(model, &params, &grads);

    // Create Adam optimizer (lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=1e-4)
    model->optimizer = adam_create(1e-3f, 0.9f, 0.999f, 1e-8f, 1e-4f);

    // Add all parameters to optimizer
    for (int i = 0; i < num_params; i++) {
        adam_add_param(model->optimizer, params[i], grads[i]);
    }

    // Create StepLR scheduler (step_size=7, gamma=0.1) - match PyTorch
    model->scheduler = step_lr_create(model->optimizer, 7, 0.1f);

    // Free parameter arrays
    cnn_free_parameters(params, grads);
}

int main() {
    printf("EMNIST Lowercase Letter CNN Training\n");
    printf("=====================================\n\n");

    // Seed random number generator
    srand(time(NULL));

    // Load EMNIST lowercase letters dataset (filtered from byclass)
    printf("Loading EMNIST Lowercase Letters dataset...\n");
    Dataset* train_dataset = dataset_load_emnist("data/font_letters_train-images.idx",
                                                "data/font_letters_train-labels.idx",
                                                64, 1, 4, NULL);  // batch_size=64, shuffle=1, num_workers=4, max_samples=NULL (use all)

    Dataset* test_dataset = dataset_load_emnist("data/font_letters_test-images.idx",
                                               "data/font_letters_test-labels.idx",
                                               1000, 0, 1, NULL);  // batch_size=1000, shuffle=0, num_workers=1, max_samples=NULL (use all)
    if (!train_dataset || !test_dataset) {
        fprintf(stderr, "Failed to load dataset\n");
        return 1;
    }

    if (!train_dataset->batches || !test_dataset->batches) {
        fprintf(stderr, "Dataset batches not allocated\n");
        return 1;
    }

    printf("Train dataset: %d batches, %d total samples\n", train_dataset->num_batches, train_dataset->total_samples);
    printf("Test dataset: %d batches, %d total samples\n\n", test_dataset->num_batches, test_dataset->total_samples);

    // Save all training images to disk
    // printf("Saving training images...\n");
    // save_all_images(train_dataset, "train_images");



    // Create model
    printf("Creating CNN model...\n");
    CNN* model = cnn_create();
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    setup_optimizer_scheduler(model);

    // Training parameters
    int num_epochs = 12;
    float best_accuracy = 0.0f;
    int patience = 10;  // More patience for deeper model
    int patience_counter = 0;

    // Save training metadata
    char dataset_info[256];
    sprintf(dataset_info, "EMNIST Lowercase Letters (filtered from byclass, %d train samples, %d test samples)",
            train_dataset->total_samples, test_dataset->total_samples);
    save_training_metadata(num_epochs, 64, 1e-3f, 1e-4f, patience, dataset_info);

    printf("Starting training for %d epochs...\n\n", num_epochs);

    // Track overall training time for ETA calculation
    double total_training_elapsed = 0.0;

    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double epoch_start_time = get_wall_time_ms();

        // Calculate overall training ETA
        char overall_eta_str[32] = "";
        if (epoch > 0) {
            double avg_time_per_epoch = total_training_elapsed / epoch;
            double remaining_epochs = num_epochs - epoch;
            double eta_seconds = remaining_epochs * avg_time_per_epoch;
            format_time_duration(eta_seconds, overall_eta_str, sizeof(overall_eta_str));
        }

        printf("Epoch %d/%d", epoch + 1, num_epochs);
        if (strlen(overall_eta_str) > 0) {
            printf(" - ETA: %s", overall_eta_str);
        }
        printf("\n");
        printf("----------\n");

        cnn_train(model);
        float epoch_loss = 0.0f;
        int total_train_samples = 0;
        int correct_train = 0;

        // Arrays to store batch data for the epoch
        float* batch_losses = (float*)malloc(train_dataset->num_batches * sizeof(float));
        float* batch_accuracies = (float*)malloc(train_dataset->num_batches * sizeof(float));
        float* batch_times = (float*)malloc(train_dataset->num_batches * sizeof(float));

        // Track total elapsed time for ETA calculation
        float total_elapsed_time = 0.0f;

        // Train on all batches
        for (int batch_idx = 0; batch_idx < train_dataset->num_batches; batch_idx++) {
            double batch_start = get_wall_time_ms();  // Start timing

            Batch* batch = &train_dataset->batches[batch_idx];

            // Normalize input from [0,1] to [-1,1] (like PyTorch transforms.Normalize((0.5,), (0.5,)))
            Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
            for (int i = 0; i < batch->data->size; i++) {
                input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;  // Normalize to [-1,1]
            }

            // Labels are already in range 0-25 (filtered and remapped by dataset loader: EMNIST 1-26 -> 0-25)
            int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
            Tensor* targets_adjusted = tensor_create(targets_shape, 4);
            for (int i = 0; i < batch->labels->size; i++) {
                targets_adjusted->data[i] = batch->labels->data[i];  // Already 0-25
            }

            // Forward pass
            CNNForwardResult* forward_result = cnn_forward(model, input_normalized);
            if (!forward_result) {
                fprintf(stderr, "Forward pass failed\n");
                // Cleanup allocated tensors
                tensor_free(input_normalized);
                tensor_free(targets_adjusted);
                // Cleanup batch arrays
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
            float batch_acc = calculate_accuracy(predictions, targets_adjusted) * 100.0f;  // Convert to percentage
            correct_train += (int)(batch_acc * batch->data->shape[0] / 100.0f);
            total_train_samples += batch->data->shape[0];
            double batch_end = get_wall_time_ms();  // End timing
            float time_per_batch = (float)(batch_end - batch_start);  // Already in milliseconds

            // Accumulate total elapsed time (convert to seconds)
            total_elapsed_time += time_per_batch / 1000.0f;

            // Store batch data for later saving
            batch_losses[batch_idx] = batch_loss;
            batch_accuracies[batch_idx] = batch_acc;
            batch_times[batch_idx] = time_per_batch;
            tensor_free(predictions);
            cnn_forward_result_free(forward_result);
            tensor_free(input_normalized);
            tensor_free(targets_adjusted);
            // Update progress bar
            float current_avg_loss = epoch_loss / total_train_samples;
            float current_avg_acc = (float)correct_train / total_train_samples * 100.0f;
            char prefix[32];
            print_progress_bar(batch_idx + 1, train_dataset->num_batches, prefix, current_avg_loss, current_avg_acc, time_per_batch, total_elapsed_time);
        }

        epoch_loss /= total_train_samples;
        float epoch_train_acc = (float)correct_train / total_train_samples * 100.0f;

        printf("\n  Train Loss: %.4f, Train Acc: %.2f%%\n", epoch_loss, epoch_train_acc);

        // Save all batch data for this epoch to a separate file
        save_epoch_batch_data(epoch + 1, batch_losses, batch_accuracies, batch_times, train_dataset->num_batches);

        // Free batch data arrays
        free(batch_losses);
        free(batch_accuracies);
        free(batch_times);

        // Track epoch time for overall ETA
        double epoch_end_time = get_wall_time_ms();
        double epoch_duration = (epoch_end_time - epoch_start_time) / 1000.0; // Convert to seconds
        total_training_elapsed += epoch_duration;

        // Step the scheduler
        cnn_step_scheduler(model);

        // Testing phase
        cnn_eval(model);
        int correct_test = 0;
        int total_test_samples = 0;

        for (int batch_idx = 0; batch_idx < test_dataset->num_batches; batch_idx++) {
            Batch* batch = &test_dataset->batches[batch_idx];

            // Normalize input from [0,1] to [-1,1]
            Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
            for (int i = 0; i < batch->data->size; i++) {
                input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;
            }

            // Labels are already in range 0-25 (filtered and remapped by dataset loader: EMNIST 1-26 -> 0-25)
            int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
            Tensor* targets_adjusted = tensor_create(targets_shape, 4);
            for (int i = 0; i < batch->labels->size; i++) {
                targets_adjusted->data[i] = batch->labels->data[i];  // Already 0-25
            }

            // Get predictions
            Tensor* predictions = cnn_predict(model, input_normalized);
            float batch_acc = calculate_accuracy(predictions, targets_adjusted);
            correct_test += (int)(batch_acc * batch->data->shape[0]);
            total_test_samples += batch->data->shape[0];

            tensor_free(predictions);
            tensor_free(input_normalized);
            tensor_free(targets_adjusted);

            // Update progress bar for testing
            float current_test_acc = (float)correct_test / total_test_samples * 100.0f;
            print_progress_bar(batch_idx + 1, test_dataset->num_batches, "Testing", 0.0f, current_test_acc, 0.0f, 0.0f);
        }

        float test_accuracy = (float)correct_test / total_test_samples * 100.0f;
        printf("  Test Accuracy: %.2f%%\n\n", test_accuracy);

        // Calculate and save additional metrics
        float grad_norm = calculate_gradient_norm(model);
        float current_lr = adam_get_learning_rate(model->optimizer);
        float lr_decay_factor = 0.1f;  // From StepLR scheduler
        save_additional_metrics(epoch + 1, current_lr, grad_norm, lr_decay_factor, patience_counter);

        // Save weights and loss data at the end of each epoch
        save_model_weights(model, epoch + 1);
        save_loss_data(epoch + 1, epoch_loss, epoch_train_acc, test_accuracy);

        // Save checkpoints
        if (test_accuracy > best_accuracy) {
            save_checkpoint(model, epoch + 1, test_accuracy, test_accuracy, "best");
        }
        // Save regular checkpoint every 5 epochs
        if ((epoch + 1) % 5 == 0) {
            save_checkpoint(model, epoch + 1, best_accuracy, test_accuracy, "regular");
        }
        // Always save latest checkpoint
        save_checkpoint(model, epoch + 1, best_accuracy, test_accuracy, "latest");

        // Early stopping check
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

    // Final testing
    printf("\nTraining completed. Best test accuracy: %.2f%%\n", best_accuracy);

    // Final evaluation
    cnn_eval(model);
    int final_correct = 0;
    int final_total = 0;

    for (int batch_idx = 0; batch_idx < test_dataset->num_batches; batch_idx++) {
        Batch* batch = &test_dataset->batches[batch_idx];

        Tensor* input_normalized = tensor_create(batch->data->shape, batch->data->ndim);
        for (int i = 0; i < batch->data->size; i++) {
            input_normalized->data[i] = (batch->data->data[i] - 0.5f) / 0.5f;
        }

        int targets_shape[] = {batch->labels->shape[0], 1, 1, 1};
        Tensor* targets_adjusted = tensor_create(targets_shape, 4);
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

        // Update progress bar for final testing
        float current_final_acc = (float)final_correct / final_total * 100.0f;
        print_progress_bar(batch_idx + 1, test_dataset->num_batches, "Final Testing", 0.0f, current_final_acc, 0.0f, 0.0f);
    }

    float final_accuracy = (float)final_correct / final_total * 100.0f;
    printf("Final Test Accuracy: %.2f%%\n", final_accuracy);

    // Cleanup
    dataset_free(train_dataset);
    dataset_free(test_dataset);
    cnn_free(model);

    return 0;
}
