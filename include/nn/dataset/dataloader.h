#pragma once

#include "nn/core/tensor.h"
#include <pthread.h>
#include <stdio.h>
#define EMNIST_IMAGE_SIZE 784  // 28x28 pixels
#define EMNIST_NUM_CLASSES 26  // A-Z (uppercase letters only)

typedef struct {
    Tensor* data;      // Batch of images (batch_size, 1, 28, 28)
    Tensor* labels;    // Batch of labels (batch_size, 1, 1, 1) - one-hot encoded or class indices
} Batch;

typedef struct {
    Batch* batches;
    int num_batches;
    int total_samples;
} Dataset;

typedef struct {
    const char* images_path;
    const char* labels_path;
    unsigned char* image_data;
    unsigned char* label_data;
    int num_images;
    int image_offset;
    int label_offset;
    int batch_size;
    int shuffle;
    int num_workers;
    int worker_id;
    int* indices;  // shuffled indices array (NULL if no shuffling)
    Dataset* dataset;
    pthread_mutex_t* mutex;
} ThreadArgs;

// IDX file format functions
int read_idx_header(FILE* file, int* magic, int* num_items, int* rows, int* cols);
unsigned char* load_idx_images(const char* path, int* num_images, int* rows, int* cols);
unsigned char* load_idx_labels(const char* path, int* num_labels);

// Multi-threading functions
void* load_batch_worker(void* args);

// Main dataset loading function
Dataset* dataset_load_emnist(const char* images_path, const char* labels_path, int batch_size, int shuffle, int num_workers, int* max_samples);
void dataset_free(Dataset* dataset);


