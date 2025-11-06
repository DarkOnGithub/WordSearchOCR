#include "../include/nn/cnn.h"
#include "../include/image/image.h"
#include "image/operations.h"
#include "../include/nn/layers/cross_entropy_loss.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <string.h>
#include <cairo/cairo.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif

// Draw text on an image using Cairo
void draw_text_on_image(Image* image, const char* text, int x, int y, const char* font_family, int font_size, double r, double g, double b) {
    if (!image || !image->rgba_pixels || !text) {
        fprintf(stderr, "Error: Invalid image or text for drawing\n");
        return;
    }

    // Convert RGBA to ARGB for Cairo
    uint32_t* argb_pixels = (uint32_t*)malloc(image->width * image->height * sizeof(uint32_t));
    if (!argb_pixels) {
        fprintf(stderr, "Error: Failed to allocate ARGB buffer for text drawing\n");
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < image->width * image->height; i++) {
        uint32_t rgba = image->rgba_pixels[i];
        uint8_t r_val = (rgba >> 24) & 0xFF;
        uint8_t g_val = (rgba >> 16) & 0xFF;
        uint8_t b_val = (rgba >> 8) & 0xFF;
        uint8_t a_val = rgba & 0xFF;
        argb_pixels[i] = (a_val << 24) | (r_val << 16) | (g_val << 8) | b_val;
    }

    cairo_surface_t* surface = cairo_image_surface_create_for_data(
        (unsigned char*)argb_pixels, CAIRO_FORMAT_ARGB32,
        image->width, image->height, image->width * 4);

    if (!surface || cairo_surface_status(surface) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to create Cairo surface for text drawing\n");
        free(argb_pixels);
        return;
    }

    cairo_t* cr = cairo_create(surface);
    if (!cr || cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Failed to create Cairo context for text drawing\n");
        cairo_surface_destroy(surface);
        free(argb_pixels);
        return;
    }

    // Set font properties
    cairo_select_font_face(cr, font_family, CAIRO_FONT_SLANT_NORMAL, CAIRO_FONT_WEIGHT_BOLD);
    cairo_set_font_size(cr, font_size);
    cairo_set_source_rgba(cr, r, g, b, 1.0);

    // Draw text
    cairo_move_to(cr, x, y);
    cairo_show_text(cr, text);

    cairo_destroy(cr);
    cairo_surface_destroy(surface);

    // Convert back to RGBA
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < image->width * image->height; i++) {
        uint32_t argb = argb_pixels[i];
        uint8_t a_val = (argb >> 24) & 0xFF;
        uint8_t r_val = (argb >> 16) & 0xFF;
        uint8_t g_val = (argb >> 8) & 0xFF;
        uint8_t b_val = argb & 0xFF;
        image->rgba_pixels[i] = (r_val << 24) | (g_val << 16) | (b_val << 8) | a_val;
    }

    free(argb_pixels);
}

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

// Check if file is an image based on extension and not already processed
int is_image_file(const char* filename) {
    // Skip files that already contain prediction results (avoid infinite loop)
    if (strstr(filename, "_pred_") != NULL) {
        return 0;
    }

    const char* ext = strrchr(filename, '.');
    if (!ext) return 0;
    return strcmp(ext, ".png") == 0 || strcmp(ext, ".jpg") == 0 ||
           strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".bmp") == 0 ||
           strcmp(ext, ".tiff") == 0 || strcmp(ext, ".tif") == 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <epoch> <folder_path>\n", argv[0]);
        fprintf(stderr, "Example: %s 3 ./test_images/\n", argv[0]);
        return 1;
    }

    int epoch = atoi(argv[1]);
    const char* folder_path = argv[2];

    if (epoch <= 0) {
        fprintf(stderr, "Error: Epoch must be a positive integer\n");
        return 1;
    }

    printf("EMNIST Letter Recognition Batch Inference\n");
    printf("=========================================\n");
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

    // Open directory
    DIR* dir = opendir(folder_path);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s\n", folder_path);
        cnn_free(model);
        return 1;
    }

    struct dirent* entry;
    int processed_count = 0;

    printf("Processing images in folder: %s\n", folder_path);
    printf("=========================================\n");

    // Create output directory for predictions
    char output_dir[1024];
    // Remove trailing slash from folder_path if present
    char clean_path[1024];
    strcpy(clean_path, folder_path);
    if (clean_path[strlen(clean_path) - 1] == '/') {
        clean_path[strlen(clean_path) - 1] = '\0';
    }
    snprintf(output_dir, sizeof(output_dir), "%s/cells_predicted", clean_path);

    // Create the directory if it doesn't exist
    if (mkdir(output_dir, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "Error: Cannot create output directory %s\n", output_dir);
        closedir(dir);
        cnn_free(model);
        return 1;
    }

    printf("Saving predictions to: %s\n", output_dir);

    // Set model to evaluation mode
    cnn_eval(model);

    // Iterate through all files in the directory
    while ((entry = readdir(dir)) != NULL) {
        // Skip hidden files and current/parent directory entries
        if (entry->d_name[0] == '.' ||
            strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        // Check if it's an image file
        if (!is_image_file(entry->d_name)) {
            continue;
        }

        // Construct full path
        char image_path[1024];
        snprintf(image_path, sizeof(image_path), "%s/%s", folder_path, entry->d_name);

        printf("Processing: %s\n", entry->d_name);

        // Load and preprocess image
        Image img;
        load_image(image_path, &img);
        if (img.width == 0 || img.height == 0) {
            fprintf(stderr, "Failed to load image: %s\n", image_path);
            continue;
        }

        // Keep original image for overlay (before preprocessing)
        Image original_img;
        cpy_image(&img, &original_img);

        // invert(&img);
        // Convert to grayscale if not already
        if (!img.is_grayscale) {
            convert_to_grayscale(&img);
        }

        // Resize to 28x28 if needed
        Image resized_img = {0};
        if (img.width != 28 || img.height != 28) {
            resize_grayscale_image(&img, &resized_img, 28, 28);
            free_image(&img);
            img = resized_img;
        }

        // Convert to tensor
        Tensor* img_tensor = to_tensor(&img);
        if (!img_tensor) {
            fprintf(stderr, "Failed to convert image to tensor\n");
            free_image(&img);
            free_image(&original_img);
            continue;
        }

        // Normalize from [0,1] to [-1,1] (same as training)
        for (int i = 0; i < img_tensor->size; i++) {
            img_tensor->data[i] = (img_tensor->data[i] - 0.5f) / 0.5f;
        }

        // Perform forward pass to get raw logits
        CNNForwardResult* forward_result = cnn_forward(model, img_tensor);
        if (!forward_result) {
            fprintf(stderr, "Forward pass failed\n");
            tensor_free(img_tensor);
            free_image(&img);
            free_image(&original_img);
            continue;
        }

        // Apply softmax to get probabilities
        Tensor* softmax_output = softmax(forward_result->fc2_out);
        if (!softmax_output) {
            fprintf(stderr, "Softmax failed\n");
            cnn_forward_result_free(forward_result);
            tensor_free(img_tensor);
            free_image(&img);
            free_image(&original_img);
            continue;
        }

        // Find the best prediction
        int predicted_class = 0;
        float max_prob = softmax_output->data[0];
        for (int c = 1; c < 26; c++) {
            if (softmax_output->data[c] > max_prob) {
                max_prob = softmax_output->data[c];
                predicted_class = c;
            }
        }

        char predicted_letter = 'A' + predicted_class +1 ;
        float percentage = max_prob * 100.0f;

        printf("  Prediction: %c (%.2f%% confidence)\n", predicted_letter, percentage);

        // Convert original image to RGBA for text overlay
        if (original_img.is_grayscale) {
            gray_to_rgba(&original_img);
        }

        // Create text to overlay
        char text[50];
        snprintf(text, sizeof(text), "%c: %.1f%%", predicted_letter, percentage);

        // Draw text on the image (bottom-left corner)
        int font_size = original_img.height / 10; // Scale font size with image height
        draw_text_on_image(&original_img, text, 10, original_img.height - 20,
                          "Sans", font_size, 1.0, 0.0, 0.0); // Red text

        // Save the processed image
        char output_path[1024];
        char base_name[256];
        strcpy(base_name, entry->d_name);
        char* dot_pos = strrchr(base_name, '.');
        if (dot_pos) *dot_pos = '\0'; // Remove extension

        snprintf(output_path, sizeof(output_path), "%s/%s_pred_%c_%.0f%%.png",
                output_dir, base_name, predicted_letter, percentage);
        save_image(output_path, &original_img);

        printf("  Saved: %s\n", output_path);

        // Cleanup
        tensor_free(softmax_output);
        cnn_forward_result_free(forward_result);
        tensor_free(img_tensor);
        free_image(&img);
        free_image(&original_img);

        processed_count++;
    }

    closedir(dir);
    cnn_free(model);

    printf("\nProcessing complete! Processed %d images.\n", processed_count);

    return 0;
}
