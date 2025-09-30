#ifndef WORDSEARCH_PROCESSOR_H
#define WORDSEARCH_PROCESSOR_H

#include "../core/image/image.h"
#include "../core/image/cv_operations.h"
/**
 * Main processing function.
 * @return 0 on success, non-zero on failure
 */
int process_wordsearch_image(const char* image_path);

/**
 * Extract individual cell images from the grid for OCR processing.
 * @return 0 on success, non-zero on failure
 */
int extract_cell_images(const Image* grid_region, const int* y_boundaries,
                       const int* x_boundaries, int num_rows, int num_cols);

/**
 * Determine grid dimensions using letter-based detection.
 * @return 0 on success, non-zero on failure
 */
int determine_grid_dimensions_from_letters(Contours* valid_letters,
                                         int crop_offset_x, int crop_offset_y,
                                         int* num_rows, int* num_cols);

/**
 * Generate cell boundaries using safe line positioning.
 * @return 0 on success, non-zero on failure
 */
int generate_safe_cell_boundaries(const Image* grid_region, int num_rows, int num_cols,
                                 int** y_boundaries, int** x_boundaries);

typedef struct {
    int row_y;
    Rect* letters;
    int count;
    int capacity;
} LetterRow;
#endif
