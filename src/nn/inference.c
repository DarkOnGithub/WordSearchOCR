#include "../include/nn/cnn.h"
#include "../include/image/image.h"
#include "image/operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Resize a grayscale image to target width and height using bilinear interpolation
void resize_grayscale_image(Image* src, Image* dst, int target_width, int target_height) {
    if (!src || !dst || !src->is_grayscale || !src->gray_pixels) {
        fprintf(stderr, "Error: Invalid source image for resizing\n");
        return;
    }

    // Allocate memory for destination image
    dst->width = target_width;
    dst->height = target_height;
    dst->is_grayscale = true;
    dst->rgba_pixels = NULL;
    dst->gray_pixels = (uint8_t*)malloc(target_width * target_height * sizeof(uint8_t));

    if (!dst->gray_pixels) {
        fprintf(stderr, "Error: Failed to allocate memory for resized image\n");
        return;
    }

    float x_ratio = (float)src->width / target_width;
    float y_ratio = (float)src->height / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            // Calculate source coordinates
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            // Get the four surrounding pixels
            int x1 = (int)floorf(src_x);
            int y1 = (int)floorf(src_y);
            int x2 = (int)ceilf(src_x);
            int y2 = (int)ceilf(src_y);

            // Clamp to image boundaries
            x1 = x1 < 0 ? 0 : (x1 >= src->width ? src->width - 1 : x1);
            x2 = x2 < 0 ? 0 : (x2 >= src->width ? src->width - 1 : x2);
            y1 = y1 < 0 ? 0 : (y1 >= src->height ? src->height - 1 : y1);
            y2 = y2 < 0 ? 0 : (y2 >= src->height ? src->height - 1 : y2);

            // Get pixel values
            uint8_t p11 = src->gray_pixels[y1 * src->width + x1];
            uint8_t p12 = src->gray_pixels[y1 * src->width + x2];
            uint8_t p21 = src->gray_pixels[y2 * src->width + x1];
            uint8_t p22 = src->gray_pixels[y2 * src->width + x2];

            // Calculate interpolation weights
            float dx = src_x - x1;
            float dy = src_y - y1;

            // Bilinear interpolation
            float interpolated = p11 * (1 - dx) * (1 - dy) +
                               p12 * dx * (1 - dy) +
                               p21 * (1 - dx) * dy +
                               p22 * dx * dy;

            // Store result
            dst->gray_pixels[y * target_width + x] = (uint8_t)roundf(interpolated);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <epoch> <image_path>\n", argv[0]);
        fprintf(stderr, "Example: %s 3 test_image.png\n", argv[0]);
        return 1;
    }

    int epoch = atoi(argv[1]);
    const char* image_path = argv[2];

    if (epoch <= 0) {
        fprintf(stderr, "Error: Epoch must be a positive integer\n");
        return 1;
    }

    printf("EMNIST Letter Recognition Inference\n");
    printf("===================================\n");
    printf("Loading model weights from epoch %d...\n", epoch);

    // Create CNN model
    CNN* model = cnn_create();
    if (!model) {
        fprintf(stderr, "Failed to create CNN model\n");
        return 1;
    }

    // Load weights from specified epoch
    if (!cnn_load_weights(model, epoch)) {
        fprintf(stderr, "Failed to load weights from epoch %d\n", epoch);
        cnn_free(model);
        return 1;
    }

    printf("Weights loaded successfully!\n");

    // Load and preprocess image
    printf("Loading image: %s\n", image_path);
    Image img;
    load_image(image_path, &img);
    if (img.width == 0 || img.height == 0) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        cnn_free(model);
        return 1;
    }
    invert(&img);
    // Convert to grayscale if not already
    if (!img.is_grayscale) {
        convert_to_grayscale(&img);
    }

    printf("Image loaded: %dx%d grayscale\n", img.width, img.height);

    // Resize to 28x28 if needed
    Image resized_img = {0};
    if (img.width != 28 || img.height != 28) {
        printf("Resizing image to 28x28...\n");
        resize_grayscale_image(&img, &resized_img, 28, 28);
        free_image(&img);  // Free original image
        img = resized_img;  // Use resized image
        printf("Image resized to 28x28\n");
    }

    // Convert to tensor
    Tensor* img_tensor = to_tensor(&img);
    if (!img_tensor) {
        fprintf(stderr, "Failed to convert image to tensor\n");
        free_image(&img);
        cnn_free(model);
        return 1;
    }
    // Reshape to NCHW format (1, 1, 28, 28)
    // The to_tensor function already creates (1, 1, 28, 28) for grayscale

    // Normalize from [0,1] to [-1,1] (same as training)
    for (int i = 0; i < img_tensor->size; i++) {
        img_tensor->data[i] = (img_tensor->data[i] - 0.5f) / 0.5f;
    }

    // Set model to evaluation mode
    cnn_eval(model);

    // Perform inference
    printf("Running inference...\n");
    Tensor* predictions = cnn_predict(model, img_tensor);
    if (!predictions) {
        fprintf(stderr, "Inference failed\n");
        tensor_free(img_tensor);
        free_image(&img);
        cnn_free(model);
        return 1;
    }

    // Get predicted class (0-25 -> A-Z)
    int predicted_class = (int)predictions->data[0];
    char predicted_letter = 'A' + predicted_class;

    printf("Prediction: %c (class %d)\n", predicted_letter, predicted_class);

    // Cleanup
    tensor_free(predictions);
    tensor_free(img_tensor);
    free_image(&img);
    cnn_free(model);

    return 0;
}
