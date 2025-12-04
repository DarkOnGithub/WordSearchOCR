#ifndef WORDSEARCH_PROCESSOR_H
#define WORDSEARCH_PROCESSOR_H

#include "../image/image.h"
#include "../image/operations.h"
#include "../solver/solver.h"

/**
 * Callback function type for creating processing step buttons
 */
typedef void (*CreateButtonCallback)(const char *step_name,
                                     const char *filename);

/**
 * Main processing function for grid detection.
 * @param image_path Path to the image to process
 * @param create_button_callback Callback function to create buttons (can be
 * NULL)
 * @param num_rows Pointer to store number of rows detected
 * @param num_cols Pointer to store number of columns detected
 * @param cell_bounding_boxes Pointer to store array of cell bounding boxes (allocated by function)
 * @param num_cells Pointer to store total number of cells
 * @param crop_offset_x Pointer to store X offset of text region in original image
 * @param crop_offset_y Pointer to store Y offset of text region in original image
 * @return 0 on success, non-zero on failure
 */
int process_wordsearch_image(const char *image_path,
                             CreateButtonCallback create_button_callback, int *num_rows, int *num_cols,
                             Rect **cell_bounding_boxes, int *num_cells,
                             int *crop_offset_x, int *crop_offset_y);

/**
 * Main processing function for word detection.
 * @param image_path Path to the image to process
 * @param create_button_callback Callback function to create buttons (can be
 * NULL)
 * @return 0 on success, non-zero on failure
 */
int process_word_detection(const char *image_path,
                           CreateButtonCallback create_button_callback);

/**
 * Extract individual cell images from the grid for OCR processing.
 * @return 0 on success, non-zero on failure
 */
int extract_cell_images(const Image *grid_region, const int *y_boundaries,
                        const int *x_boundaries, int num_rows, int num_cols);

/**
 * Determine grid dimensions using letter-based detection.
 * @return 0 on success, non-zero on failure
 */
int determine_grid_dimensions_from_letters(Contours *valid_letters,
                                           int crop_offset_x, int crop_offset_y,
                                           int *num_rows, int *num_cols);

/**
 * Generate cell boundaries using safe line positioning.
 * @return 0 on success, non-zero on failure
 */
int generate_safe_cell_boundaries(const Image *grid_region, int num_rows,
                                  int num_cols, int **y_boundaries,
                                  int **x_boundaries);

/**
 * Generate cell boundaries based on letter positions and gaps.
 * @return 0 on success, non-zero on failure
 */
int generate_cell_boundaries_from_letters(Contours *valid_letters, int num_rows,
                                          int num_cols, int **y_boundaries,
                                          int **x_boundaries);

/**
 * Create a reconstructed grid image from individual cell images with spacing.
 * @return 0 on success, non-zero on failure
 */
int create_reconstructed_grid(int num_rows, int num_cols,
                              CreateButtonCallback create_button_callback);

typedef struct
{
    int row_y;
    Rect *letters;
    int count;
    int capacity;
} LetterRow;

/**
 * Draw capsules around solved words on the original image.
 * @param image_path Path to the original image
 * @param word_matches Array of word matches to draw
 * @param num_matches Number of word matches
 * @param num_rows Number of rows in the grid
 * @param num_cols Number of columns in the grid
 * @param cell_bounding_boxes Cell bounding boxes (relative to text region)
 * @param text_region_offset_x X offset of text region in original image
 * @param text_region_offset_y Y offset of text region in original image
 * @param output_path Path to save the annotated image (can be same as input)
 * @return 0 on success, non-zero on failure
 */
int draw_solved_words(const char *image_path, WordMatch **word_matches, int num_matches,
                      WordsArray *words_array,
                      int num_rows, int num_cols, const Rect *cell_bounding_boxes,
                      int text_region_offset_x, int text_region_offset_y, const char *output_path);
#endif
