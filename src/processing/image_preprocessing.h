#ifndef IMAGE_PREPROCESSING_H
#define IMAGE_PREPROCESSING_H

#include "../core/image/image.h"
#include "../core/image/cv_operations.h"
/**
 * Load an image from file and perform initial preprocessing.
 * @return 1 on success, 0 on failure
 */
int load_and_preprocess_image(const char* image_path, Image* image);

/**
 * Extract the grid region from an image using contour detection.
 * @return 1 on success, 0 on failure
 */
int extract_grid_region(const Image* processed_image, const Image* original_image, Image* grid_image, Rect* grid_bounds);

/**
 * Process the grid region for OCR - apply thresholding and morphological operations.
 * @return 1 on success, 0 on failure
 */
int process_grid_for_ocr(Image* grid_image);

/**
 * Determine the optimal text region based on letter contours.
 * @return 1 on success, 0 on failure
 */
int determine_text_region(const Image* original_image, int grid_x, int grid_y,
                         int grid_width, int grid_height, const Contours* valid_letters,
                         Image* text_region, int* crop_offset_x, int* crop_offset_y);

#endif
