#include "preprocessing.h"
#include "../analysis/contour_analysis.h"
#include <stdio.h>

int load_and_preprocess_image(const char* image_path, Image* image, CreateButtonCallback create_button_callback) {
    if (!image_path || !image) {
        return 0;
    }

    load_image(image_path, image);
    if (!image) {
        printf("Failed to load image: %s\n", image_path);
        return 0;
    }

    convert_to_grayscale(image);
    save_image("step_01_grayscale.png", image);
    if (create_button_callback) {
        create_button_callback("Grayscale", "step_01_grayscale.png");
    }

    adaptive_denoise(image);
    save_image("step_02_adaptive_denoise.png", image);
    if (create_button_callback) {
        create_button_callback("Adaptive Denoise", "step_02_adaptive_denoise.png");
    }

    adaptiveThreshold(image, 255, 1, 1, 11, 2.0);
    save_image("step_03_threshold.png", image);
    if (create_button_callback) {
        create_button_callback("Threshold", "step_03_threshold.png");
    }

    adaptive_morphological_clean(image);
    save_image("step_03_5_morph_cleaned.png", image);
    if (create_button_callback) {
        create_button_callback("Morphological Clean", "step_03_5_morph_cleaned.png");
    }

    return 1;
}

int extract_grid_region(const Image* processed_image, const Image* original_image, Image* grid_image, Rect* grid_bounds, CreateButtonCallback create_button_callback) {
    if (!processed_image || !original_image || !grid_image) {
        return 0;
    }

    // Find contours in the processed binary image
    Contours* grid_contours = findContours(processed_image, 0);
    if (!grid_contours || grid_contours->count == 0) {
        printf("No contours found in processed image\n");
        if (grid_contours) freeContours(grid_contours);
        return 0;
    }

    // Find the largest contour (assumed to be the grid boundary)
    Contour* best_contour = find_largest_contour(grid_contours);
    if (!best_contour) {
        printf("Failed to find largest contour\n");
        freeContours(grid_contours);
        return 0;
    }

    // Get bounding rectangle of the grid
    if (!boundingRect(best_contour, grid_bounds)) {
        printf("Failed to get bounding rectangle for grid contour\n");
        freeContours(grid_contours);
        return 0;
    }

    int x = grid_bounds->x, y = grid_bounds->y;
    int w = grid_bounds->width, h = grid_bounds->height;
    int area = w * h;

    printf("Grid contour bounding rect: x=%d, y=%d, w=%d, h=%d (area=%d)\n",
           x, y, w, h, area);

    extract_rectangle(original_image, x, y, w, h, grid_image);
    convert_to_grayscale(grid_image);
    save_image("step_05_grid_extraction.png", grid_image);
    if (create_button_callback) {
        create_button_callback("Grid Extraction", "step_05_grid_extraction.png");
    }

    freeContours(grid_contours);

    return 1;
}

int process_grid_for_ocr(Image* grid_image, CreateButtonCallback create_button_callback) {
    if (!grid_image) {
        return 0;
    }

    double otsu_threshold = threshold(grid_image, 255);
    if (otsu_threshold < 0) {
        printf("Failed to apply Otsu's thresholding\n");
        return 0;
    }

    if (correctBinaryImageOrientation(grid_image) < 0) {
        printf("Failed to correct binary image orientation\n");
        return 0;
    }

    save_image("step_06_binary_grid.png", grid_image);
    if (create_button_callback) {
        create_button_callback("Binary Grid", "step_06_binary_grid.png");
    }

    StructuringElement* cleanup_kernel = getStructuringElement(0, 2, 2);
    if (cleanup_kernel) {
        morphologyEx(grid_image, MORPH_CLOSE, cleanup_kernel, 1);
        freeStructuringElement(cleanup_kernel);
        save_image("step_07_cleaned_grid.png", grid_image);
        if (create_button_callback) {
            create_button_callback("Cleaned Grid", "step_07_cleaned_grid.png");
        }
    }

    return 1;
}

int determine_text_region(const Image* original_image, int grid_x, int grid_y,
                         int grid_width, int grid_height, const Contours* valid_letters,
                         Image* text_region, int* crop_offset_x, int* crop_offset_y) {
    if (!original_image || !text_region) {
        return 0;
    }

    if (!valid_letters || valid_letters->count == 0) {
        // No letters found, use full grid region
        extract_rectangle(original_image, grid_x, grid_y, grid_width, grid_height, text_region);
        convert_to_grayscale(text_region);
        if (crop_offset_x) *crop_offset_x = 0;
        if (crop_offset_y) *crop_offset_y = 0;
        printf("No letters found, using full grid region\n");
        return 1;
    }

    Rect* letter_rects = (Rect*)malloc(sizeof(Rect) * valid_letters->count);
    if (!letter_rects) {
        printf("Failed to allocate memory for letter rectangles\n");
        return 0;
    }

    for (int i = 0; i < valid_letters->count; i++) {
        if (!boundingRect(&valid_letters->contours[i], &letter_rects[i])) {
            printf("Failed to get bounding rectangle for letter %d\n", i);
            free(letter_rects);
            return 0;
        }
    }

    Rect text_bounds;
    if (getBoundingRectOfRects(letter_rects, valid_letters->count, 5,
                              grid_width, grid_height, &text_bounds)) {
        int translated_x = grid_x + text_bounds.x;
        int translated_y = grid_y + text_bounds.y;

        extract_rectangle(original_image, translated_x, translated_y,
                         text_bounds.width, text_bounds.height, text_region);
        convert_to_grayscale(text_region);

        if (crop_offset_x) *crop_offset_x = text_bounds.x;
        if (crop_offset_y) *crop_offset_y = text_bounds.y;

        printf("Letter-based detection: cropped text region: %dx%d at (%d,%d) [translated from grid coords (%d,%d)]\n",
               text_bounds.width, text_bounds.height, translated_x, translated_y,
               text_bounds.x, text_bounds.y);
    } else {
        printf("Failed to calculate text bounds, using full grid region\n");
        extract_rectangle(original_image, grid_x, grid_y, grid_width, grid_height, text_region);
        convert_to_grayscale(text_region);
        if (crop_offset_x) *crop_offset_x = 0;
        if (crop_offset_y) *crop_offset_y = 0;
    }

    free(letter_rects);
    return 1;
}
