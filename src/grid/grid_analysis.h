#ifndef GRID_ANALYSIS_H
#define GRID_ANALYSIS_H

#include "../core/image/image.h"

/**
 * Determine if the grid has proper grid lines that require line-based cell detection.
 * This is done by detecting horizontal and vertical lines that span nearly the full width/height.
 * @return 1 if proper grid lines detected (use line detection), 0 if not (use letter-based detection)
 */
int has_proper_grid_lines(const Image* grid_region);

/**
 * Detect grid lines using morphological operations.
 */
void detect_grid_lines(const Image* binary_grid, Image* horizontal_lines, Image* vertical_lines);

/**
 * Extract cell boundaries from detected lines.
 * @return 1 on success, 0 on failure
 */
int extract_cell_boundaries_from_lines(const Image* horizontal_lines, const Image* vertical_lines,
                                      int grid_width, int grid_height,
                                      int** y_boundaries, int** x_boundaries,
                                      int* num_rows, int* num_cols);

#endif