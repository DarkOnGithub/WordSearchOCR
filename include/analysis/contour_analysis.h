#ifndef CONTOUR_ANALYSIS_H
#define CONTOUR_ANALYSIS_H

#include "../image/image.h"
#include "../image/operations.h"

/**
 * Find the contour with the largest area from a set of contours.
 * @return Pointer to the largest contour, or NULL if no contours found
 */
Contour *find_largest_contour(const Contours *contours);

/**
 * Filter contours that could represent letters based on area and aspect ratio
 * criteria. Criteria: 20 < area < 5000, 0.3 < w/h < 3.0, w > 8, h > 8
 * @return New Contours structure with filtered contours, or NULL on error
 */
Contours *filter_letter_contours(const Contours *contours);

/**
 * Translate contour coordinates by an offset.
 */
void translateContours(Contours *contours, int offset_x, int offset_y);

/**
 * Find a safe line position that avoids overlapping with letters.
 * Tries the target position first, then shifts up/down or left/right within
 * max_offset.
 * @return Safe position that doesn't overlap letters, or original target_pos if
 * none found
 */
int find_safe_line_position(int target_pos, int is_horizontal, int max_offset,
                            const Image *binary_img, int img_width,
                            int img_height);

#endif
