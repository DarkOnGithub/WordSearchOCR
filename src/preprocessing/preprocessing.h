#ifndef IMAGE_PREPROCESSING_H
#define IMAGE_PREPROCESSING_H

#include "../image/image.h"
#include "../image/operations.h"

/**
 * Callback function type for creating processing step buttons
 */
typedef void (*CreateButtonCallback)(const char *step_name,
                                     const char *filename);

/**
 * Load an image from file and perform initial preprocessing.
 * @param image_path Path to the image file
 * @param image Output image structure
 * @param create_button_callback Callback function to create buttons (can be
 * NULL)
 * @return 1 on success, 0 on failure
 */
int load_and_preprocess_image(const char *image_path, Image *image,
                              CreateButtonCallback create_button_callback);

/**
 * Extract the grid region from an image using contour detection.
 * @param processed_image The preprocessed image
 * @param original_image The original image
 * @param grid_image Output grid image
 * @param grid_bounds Output grid bounds
 * @param create_button_callback Callback function to create buttons (can be
 * NULL)
 * @return 1 on success, 0 on failure
 */
int extract_grid_region(const Image *processed_image,
                        const Image *original_image, Image *grid_image,
                        Rect *grid_bounds,
                        CreateButtonCallback create_button_callback);

/**
 * Process the grid region for OCR - apply thresholding and morphological
 * operations.
 * @param grid_image The grid image to process
 * @param create_button_callback Callback function to create buttons (can be
 * NULL)
 * @return 1 on success, 0 on failure
 */
int process_grid_for_ocr(Image *grid_image,
                         CreateButtonCallback create_button_callback);

/**
 * Determine the optimal text region based on letter contours.
 * @return 1 on success, 0 on failure
 */
int determine_text_region(const Image *original_image, int grid_x, int grid_y,
                          int grid_width, int grid_height,
                          const Contours *valid_letters, Image *text_region,
                          int *crop_offset_x, int *crop_offset_y);

#endif
