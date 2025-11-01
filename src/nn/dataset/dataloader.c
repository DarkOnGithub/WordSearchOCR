#include "nn/dataset/dataloader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// Read IDX file header
int read_idx_header(FILE* file, int* magic, int* num_items, int* rows, int* cols) {
    if (fread(magic, sizeof(int), 1, file) != 1) return 0;
    *magic = reverse_int(*magic);

    if (fread(num_items, sizeof(int), 1, file) != 1) return 0;
    *num_items = reverse_int(*num_items);

    // For labels (1D data), rows and cols are not present in header
    if (*magic == 2049) {  // Labels magic number
        *rows = 0;
        *cols = 0;
        return 1;
    }

    // For images (3D data), read dimensions
    if (*magic == 2051) {  // Images magic number
        if (fread(rows, sizeof(int), 1, file) != 1) return 0;
        *rows = reverse_int(*rows);

        if (fread(cols, sizeof(int), 1, file) != 1) return 0;
        *cols = reverse_int(*cols);
        return 1;
    }

    // Unknown magic number
    return 0;
}

// Load IDX images
unsigned char* load_idx_images(const char* path, int* num_images, int* rows, int* cols) {
    printf("  Opening image file: %s\n", path);
    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("  Error: Cannot open file\n");
        return NULL;
    }

    int magic;
    printf("  Reading header...\n");
    if (!read_idx_header(file, &magic, num_images, rows, cols)) {
        printf("  Error: Failed to read header\n");
        fclose(file);
        return NULL;
    }

    printf("  Magic: 0x%08X, Images: %d, Size: %dx%d\n", magic, *num_images, *rows, *cols);

    // Verify magic number for images (0x00000803)
    if (magic != 2051) {
        printf("  Error: Invalid magic number for images (expected 2051, got %d)\n", magic);
        fclose(file);
        return NULL;
    }

    size_t image_size = (*rows) * (*cols);
    size_t total_size = (*num_images) * image_size;
    printf("  Allocating %zu bytes for image data...\n", total_size);
    unsigned char* data = (unsigned char*)malloc(total_size);

    if (!data) {
        printf("  Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    printf("  Reading image data...\n");
    size_t bytes_read = fread(data, sizeof(unsigned char), total_size, file);
    if (bytes_read != total_size) {
        printf("  Error: Read %zu bytes, expected %zu\n", bytes_read, total_size);
        free(data);
        fclose(file);
        return NULL;
    }

    printf("  Successfully loaded image data\n");
    fclose(file);
    return data;
}

// Load IDX labels
unsigned char* load_idx_labels(const char* path, int* num_labels) {
    printf("  Opening label file: %s\n", path);
    FILE* file = fopen(path, "rb");
    if (!file) {
        printf("  Error: Cannot open file\n");
        return NULL;
    }

    int magic, rows, cols;
    printf("  Reading header...\n");
    if (!read_idx_header(file, &magic, num_labels, &rows, &cols)) {
        printf("  Error: Failed to read header\n");
        fclose(file);
        return NULL;
    }

    printf("  Magic: 0x%08X, Labels: %d\n", magic, *num_labels);

    // Verify magic number for labels (0x00000801)
    if (magic != 2049) {
        printf("  Error: Invalid magic number for labels (expected 2049, got %d)\n", magic);
        fclose(file);
        return NULL;
    }

    printf("  Allocating %d bytes for label data...\n", *num_labels);
    unsigned char* data = (unsigned char*)malloc(*num_labels);
    if (!data) {
        printf("  Error: Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    printf("  Reading label data...\n");
    size_t bytes_read = fread(data, sizeof(unsigned char), *num_labels, file);
    if (bytes_read != (size_t)*num_labels) {
        printf("  Error: Read %zu bytes, expected %d\n", bytes_read, *num_labels);
        free(data);
        fclose(file);
        return NULL;
    }

    printf("  Successfully loaded label data\n");
    fclose(file);
    return data;
}

// Worker function for multi-threaded batch loading
void* load_batch_worker(void* args) {
    ThreadArgs* thread_args = (ThreadArgs*)args;
    int batch_idx = thread_args->worker_id;
    int start_sample = batch_idx * thread_args->batch_size;
    int end_sample = (start_sample + thread_args->batch_size > thread_args->num_images) ?
                     thread_args->num_images : start_sample + thread_args->batch_size;

    int actual_batch_size = end_sample - start_sample;
    if (actual_batch_size <= 0) return NULL;

    // Create batch tensors
    int data_shape[4] = {actual_batch_size, 1, 28, 28};
    int label_shape[3] = {actual_batch_size, 1, 1};
    thread_args->dataset->batches[batch_idx].data = tensor_create(data_shape, 4);
    thread_args->dataset->batches[batch_idx].labels = tensor_create(label_shape, 3);

    if (!thread_args->dataset->batches[batch_idx].data || !thread_args->dataset->batches[batch_idx].labels) {
        return NULL;
    }

    // Load data into tensors
    for (int i = 0; i < actual_batch_size; i++) {
        // Use shuffled indices if available, otherwise sequential
        int sample_idx = thread_args->indices ? thread_args->indices[start_sample + i] : (start_sample + i);

        // Copy image data (normalize to 0-1 range)
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int file_pixel_idx = sample_idx * EMNIST_IMAGE_SIZE + y * 28 + x;
                int tensor_pixel_idx = y * 28 + x;
                int tensor_idx = i * EMNIST_IMAGE_SIZE + tensor_pixel_idx;

                thread_args->dataset->batches[batch_idx].data->data[tensor_idx] =
                    (float)thread_args->image_data[file_pixel_idx] / 255.0f;
            }
        }

        // Set label (as class index, not one-hot for now)
        thread_args->dataset->batches[batch_idx].labels->data[i] =
            (float)thread_args->label_data[sample_idx];
    }

    return NULL;
}


// EMNIST dataset loading function with sample limit for testing
Dataset* dataset_load_emnist(const char* images_path, const char* labels_path, int batch_size, int shuffle, int num_workers, int* max_samples) {
    printf("Loading images from: %s\n", images_path);
    printf("Loading labels from: %s\n", labels_path);
    if (max_samples) {
        printf("Limiting to %d samples for testing\n", *max_samples);
    } else {
        printf("Using all available samples\n");
    }

    // Load raw data
    int num_images, rows, cols, num_labels;
    unsigned char* image_data = load_idx_images(images_path, &num_images, &rows, &cols);

    if (!image_data) {
        printf("Error: Failed to load image data\n");
        return NULL;
    }

    printf("Loaded %d images with dimensions %dx%d\n", num_images, rows, cols);

    unsigned char* label_data = load_idx_labels(labels_path, &num_labels);

    if (!label_data) {
        printf("Error: Failed to load label data\n");
        free(image_data);
        return NULL;
    }

    printf("Loaded %d labels\n", num_labels);

    if (num_images != num_labels) {
        printf("Error: Number of images (%d) doesn't match number of labels (%d)\n", num_images, num_labels);
        free(image_data);
        free(label_data);
        return NULL;
    }

    if (rows != 28 || cols != 28) {
        printf("Error: Expected 28x28 images, got %dx%d\n", rows, cols);
        free(image_data);
        free(label_data);
        return NULL;
    }

    // EMNIST Letters dataset should have classes 1-26 (A-Z)
    // Count valid samples and check class range
    int min_class = 27, max_class = -1;
    int filtered_count = 0;
    for (int i = 0; i < num_images; i++) {
        int label = (int)label_data[i];
        if (label >= 1 && label <= 26) {  // uppercase letters A-Z
            filtered_count++;
            if (label < min_class) min_class = label;
            if (label > max_class) max_class = label;
        }
    }

    printf("Original dataset: %d samples\n", num_images);
    printf("Filtered dataset (uppercase letters A-Z only): %d samples\n", filtered_count);
    printf("Class range: %d-%d\n", min_class, max_class);

    if (filtered_count == 0) {
        printf("Error: No valid samples found after filtering\n");
        free(image_data);
        free(label_data);
        return NULL;
    }

    // Check that we have exactly 26 classes (A-Z)
    if (min_class != 1 || max_class != 26) {
        printf("Error: Expected classes 1-26 (A-Z), but found range %d-%d\n", min_class, max_class);
        free(image_data);
        free(label_data);
        return NULL;
    }

    printf("✓ Confirmed: Dataset contains exactly 26 classes (A-Z)\n");

    // Limit samples if max_samples is specified and positive
    int final_sample_count = filtered_count;
    if (max_samples && *max_samples > 0 && *max_samples < filtered_count) {
        final_sample_count = *max_samples;
        printf("Limiting dataset to %d samples as requested\n", final_sample_count);
    }

    // Create filtered arrays with final sample count
    unsigned char* filtered_images = (unsigned char*)malloc(final_sample_count * EMNIST_IMAGE_SIZE);
    unsigned char* filtered_labels = (unsigned char*)malloc(final_sample_count);

    if (!filtered_images || !filtered_labels) {
        printf("Error: Failed to allocate filtered arrays\n");
        free(image_data);
        free(label_data);
        return NULL;
    }

    int filtered_idx = 0;
    for (int i = 0; i < num_images; i++) {
        int label = (int)label_data[i];
        if (label >= 1 && label <= 26 && filtered_idx < final_sample_count) {
            memcpy(&filtered_images[filtered_idx * EMNIST_IMAGE_SIZE],
                   &image_data[i * EMNIST_IMAGE_SIZE],
                   EMNIST_IMAGE_SIZE);

            filtered_labels[filtered_idx] = label - 1;  // Convert 1-26 (A-Z) to 0-25

            filtered_idx++;
        }
    }

    // Replace original arrays with filtered ones
    free(image_data);
    free(label_data);
    image_data = filtered_images;
    label_data = filtered_labels;
    num_images = final_sample_count;

    // Calculate number of batches
    int num_batches = (num_images + batch_size - 1) / batch_size;  // Ceiling division

    // Allocate dataset
    Dataset* dataset = (Dataset*)malloc(sizeof(Dataset));
    if (!dataset) {
        printf("Error: Failed to allocate dataset\n");
        free(image_data);
        free(label_data);
        return NULL;
    }

    dataset->batches = (Batch*)malloc(sizeof(Batch) * num_batches);
    if (!dataset->batches) {
        printf("Error: Failed to allocate batches\n");
        free(dataset);
        free(image_data);
        free(label_data);
        return NULL;
    }

    dataset->num_batches = num_batches;
    dataset->total_samples = num_images;

    // Shuffle indices if requested
    int* indices = NULL;
    if (shuffle) {
        indices = (int*)malloc(sizeof(int) * num_images);
        if (!indices) {
            printf("Error: Failed to allocate indices\n");
            free(dataset);
            free(image_data);
            free(label_data);
            return NULL;
        }
        for (int i = 0; i < num_images; i++) indices[i] = i;
        // Simple Fisher-Yates shuffle
        for (int i = num_images - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    // Multi-threaded batch loading
    pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_batches);
    ThreadArgs* thread_args = (ThreadArgs*)malloc(sizeof(ThreadArgs) * num_batches);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

    // Create all batch processing threads
    for (int b = 0; b < num_batches; b++) {
        thread_args[b] = (ThreadArgs){
            .images_path = images_path,
            .labels_path = labels_path,
            .image_data = image_data,
            .label_data = label_data,
            .num_images = num_images,
            .batch_size = batch_size,
            .shuffle = shuffle,
            .num_workers = num_workers,
            .worker_id = b,  // batch index
            .indices = indices,  // shuffled indices (or NULL)
            .dataset = dataset,
            .mutex = &mutex
        };

        pthread_create(&threads[b], NULL, load_batch_worker, &thread_args[b]);
    }

    // Wait for all threads to complete
    for (int b = 0; b < num_batches; b++) {
        pthread_join(threads[b], NULL);
    }

    // Cleanup
    free(threads);
    free(thread_args);
    free(image_data);
    free(label_data);
    if (indices) free(indices);

    printf("✓ Successfully loaded subset dataset with %d samples\n", num_images);
    return dataset;
}

void dataset_free(Dataset* dataset) {
    if (!dataset) return;

    for (int i = 0; i < dataset->num_batches; i++) {
        if (dataset->batches[i].data) tensor_free(dataset->batches[i].data);
        if (dataset->batches[i].labels) tensor_free(dataset->batches[i].labels);
    }

    free(dataset->batches);
    free(dataset);
}