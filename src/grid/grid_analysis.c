#include "./grid_analysis.h"
#include "../core/image/cv_operations.h"
#include <stdlib.h>
#include <stdio.h>


int has_proper_grid_lines(const Image* grid_region) {
    if (!grid_region || grid_region->width <= 0 || grid_region->height <= 0) {
        return 0;
    }

    Image horizontal_lines = {0};
    Image vertical_lines = {0};
    detect_grid_lines(grid_region, &horizontal_lines, &vertical_lines);

    if (horizontal_lines.width <= 0 || horizontal_lines.height <= 0 ||
        vertical_lines.width <= 0 || vertical_lines.height <= 0) {
        free_image(&horizontal_lines);
        free_image(&vertical_lines);
        return 0;
    }

    // Find contours in the detected lines
    Contours* horiz_contours = findContours(&horizontal_lines, 0);
    Contours* vert_contours = findContours(&vertical_lines, 0);

    free_image(&horizontal_lines);
    free_image(&vertical_lines);

    if (!horiz_contours || !vert_contours) {
        if (horiz_contours) freeContours(horiz_contours);
        if (vert_contours) freeContours(vert_contours);
        return 0;
    }

    // Filter contours by length - they should span at least 60% of the grid width/height
    double min_horiz_length = grid_region->width * 0.6;
    double min_vert_length = grid_region->height * 0.6;

    Contours* filtered_horiz = filterContoursByLength(horiz_contours, min_horiz_length);
    Contours* filtered_vert = filterContoursByLength(vert_contours, min_vert_length);

    freeContours(horiz_contours);
    freeContours(vert_contours);

    if (!filtered_horiz || !filtered_vert) {
        if (filtered_horiz) freeContours(filtered_horiz);
        if (filtered_vert) freeContours(filtered_vert);
        return 0;
    }

    int horiz_line_count = filtered_horiz->count;
    int vert_line_count = filtered_vert->count;

    freeContours(filtered_horiz);
    freeContours(filtered_vert);

    printf("Grid line detection: found %d horizontal lines, %d vertical lines\n",
           horiz_line_count, vert_line_count);

    // If we found at least 3 lines in each direction that span the full grid, assume we have proper grid lines
    if (horiz_line_count >= 3 && vert_line_count >= 3) {
        printf("Detected proper grid lines - will use line detection\n");
        return 1;
    } else {
        printf("No proper grid lines detected - will use letter-based detection\n");
        return 0;
    }
}

void detect_grid_lines(const Image* binary_grid, Image* horizontal_lines, Image* vertical_lines) {
    // Use adaptive kernels for different line thicknesses
    int min_dim = binary_grid->width < binary_grid->height ?
        binary_grid->width : binary_grid->height;
    int kernel_length = min_dim / 30; // Adaptive kernel size based on image size
    if (kernel_length < 5) kernel_length = 5;   // Very small minimum for thin lines
    if (kernel_length > 40) kernel_length = 40; // Maximum size

    printf("Using morphological kernel length: %d\n", kernel_length);

    // For thin lines, use thinner kernels (1 pixel thick)
    int thickness = 1;
    StructuringElement* horiz_kernel = getStructuringElement(0, kernel_length, thickness); // MORPH_RECT, length x thickness
    if (!horiz_kernel) {
        printf("Failed to create horizontal line kernel\n");
        return;
    }

    StructuringElement* vert_kernel = getStructuringElement(0, thickness, kernel_length); // MORPH_RECT, thickness x length
    if (!vert_kernel) {
        printf("Failed to create vertical line kernel\n");
        freeStructuringElement(horiz_kernel);
        return;
    }

    cpy_image(binary_grid, horizontal_lines);
    cpy_image(binary_grid, vertical_lines);

    // Detect horizontal lines: erode with horizontal kernel, then dilate
    morphologyEx(horizontal_lines, MORPH_ERODE, horiz_kernel, 1); // MORPH_ERODE
    morphologyEx(horizontal_lines, MORPH_DILATE, horiz_kernel, 1); // MORPH_DILATE

    // Detect vertical lines: erode with vertical kernel, then dilate
    morphologyEx(vertical_lines, MORPH_ERODE, vert_kernel, 1); // MORPH_ERODE
    morphologyEx(vertical_lines, MORPH_DILATE, vert_kernel, 1); // MORPH_DILATE

    freeStructuringElement(horiz_kernel);
    freeStructuringElement(vert_kernel);
}

int extract_cell_boundaries_from_lines(const Image* horizontal_lines, const Image* vertical_lines,
                                      int grid_width, int grid_height,
                                      int** y_boundaries, int** x_boundaries,
                                      int* num_rows, int* num_cols) {
    Contours* horiz_contours = findContours(horizontal_lines, 0);
    if (!horiz_contours) {
        printf("Failed to find horizontal line contours\n");
        return 0;
    }

    Contours* vert_contours = findContours(vertical_lines, 0);
    if (!vert_contours) {
        printf("Failed to find vertical line contours\n");
        freeContours(horiz_contours);
        return 0;
    }


    Contours* filtered_horiz = filterContoursByLength(horiz_contours, grid_width * 0.2);
    freeContours(horiz_contours);

    Contours* filtered_vert = filterContoursByLength(vert_contours, grid_height * 0.2);
    freeContours(vert_contours);

    if (!filtered_horiz || !filtered_vert) {
        if (filtered_horiz) freeContours(filtered_horiz);
        if (filtered_vert) freeContours(filtered_vert);
        return 0;
    }

    int detected_horiz = filtered_horiz->count;
    int detected_vert = filtered_vert->count;

    if (detected_horiz < 3 || detected_vert < 3) {
        freeContours(filtered_horiz);
        freeContours(filtered_vert);
        return 0;
    }

    int grid_size_lines = detected_horiz < detected_vert ? detected_horiz : detected_vert;
    //!WARNING: This is a hardcoded value for the max grid size.
    if (grid_size_lines > 21) grid_size_lines = 21;

    *num_rows = grid_size_lines - 1;
    *num_cols = grid_size_lines - 1;

    *y_boundaries = (int*)malloc(sizeof(int) * grid_size_lines);
    *x_boundaries = (int*)malloc(sizeof(int) * grid_size_lines);
    if (!*y_boundaries || !*x_boundaries) {
        if (*y_boundaries) free(*y_boundaries);
        if (*x_boundaries) free(*x_boundaries);
        freeContours(filtered_horiz);
        freeContours(filtered_vert);
        return 0;
    }

    // Sort and extract Y boundaries from horizontal lines
    if (filtered_horiz->count > 0) {
        for (int i = 0; i < filtered_horiz->count - 1; i++) {
            for (int j = 0; j < filtered_horiz->count - i - 1; j++) {
                Rect rect1, rect2;
                boundingRect(&filtered_horiz->contours[j], &rect1);
                boundingRect(&filtered_horiz->contours[j + 1], &rect2);

                if (rect1.y > rect2.y) {
                    Contour temp = filtered_horiz->contours[j];
                    filtered_horiz->contours[j] = filtered_horiz->contours[j + 1];
                    filtered_horiz->contours[j + 1] = temp;
                }
            }
        }

        // Extract Y boundaries - distribute evenly if we have fewer than expected
        if (filtered_horiz->count >= grid_size_lines) {
            for (int i = 0; i < grid_size_lines; i++) {
                Rect rect;
                boundingRect(&filtered_horiz->contours[i], &rect);
                (*y_boundaries)[i] = rect.y;
            }
        } else {
            // If we have fewer lines than expected, interpolate
            for (int i = 0; i < filtered_horiz->count; i++) {
                Rect rect;
                boundingRect(&filtered_horiz->contours[i], &rect);
                int target_idx = (int)((float)i / (filtered_horiz->count - 1) * (grid_size_lines - 1));
                if (target_idx < grid_size_lines) {
                    (*y_boundaries)[target_idx] = rect.y;
                }
            }
            // Fill in missing boundaries by interpolation
            for (int i = 0; i < grid_size_lines; i++) {
                if (i > 0 && (*y_boundaries)[i] == 0) {
                    // Interpolate between neighboring known boundaries
                    int prev_known = -1, next_known = -1;
                    for (int j = i - 1; j >= 0; j--) {
                        if ((*y_boundaries)[j] != 0) {
                            prev_known = j;
                            break;
                        }
                    }
                    for (int j = i + 1; j < grid_size_lines; j++) {
                        if ((*y_boundaries)[j] != 0) {
                            next_known = j;
                            break;
                        }
                    }
                    if (prev_known >= 0 && next_known >= 0) {
                        float ratio = (float)(i - prev_known) / (next_known - prev_known);
                        (*y_boundaries)[i] = (*y_boundaries)[prev_known] +
                                           (int)(ratio * ((*y_boundaries)[next_known] - (*y_boundaries)[prev_known]));
                    }
                }
            }
        }
    } else {
        // No horizontal lines detected, create evenly spaced boundaries
        for (int i = 0; i < grid_size_lines; i++) {
            (*y_boundaries)[i] = (i * grid_height) / (*num_rows);
        }
    }

    // Sort and extract X boundaries from vertical lines
    if (filtered_vert->count > 0) {
        for (int i = 0; i < filtered_vert->count - 1; i++) {
            for (int j = 0; j < filtered_vert->count - i - 1; j++) {
                Rect rect1, rect2;
                boundingRect(&filtered_vert->contours[j], &rect1);
                boundingRect(&filtered_vert->contours[j + 1], &rect2);

                if (rect1.x > rect2.x) {
                    Contour temp = filtered_vert->contours[j];
                    filtered_vert->contours[j] = filtered_vert->contours[j + 1];
                    filtered_vert->contours[j + 1] = temp;
                }
            }
        }

        // Extract X boundaries - distribute evenly if we have fewer than expected
        if (filtered_vert->count >= grid_size_lines) {
            // If we have enough lines, use the first grid_size_lines
            for (int i = 0; i < grid_size_lines; i++) {
                Rect rect;
                boundingRect(&filtered_vert->contours[i], &rect);
                (*x_boundaries)[i] = rect.x;
            }
        } else {
            // If we have fewer lines than expected, interpolate
            for (int i = 0; i < filtered_vert->count; i++) {
                Rect rect;
                boundingRect(&filtered_vert->contours[i], &rect);
                int target_idx = (int)((float)i / (filtered_vert->count - 1) * (grid_size_lines - 1));
                if (target_idx < grid_size_lines) {
                    (*x_boundaries)[target_idx] = rect.x;
                }
            }
            // Fill in missing boundaries by interpolation
            for (int i = 0; i < grid_size_lines; i++) {
                if (i > 0 && (*x_boundaries)[i] == 0) {
                    // Interpolate between neighboring known boundaries
                    int prev_known = -1, next_known = -1;
                    for (int j = i - 1; j >= 0; j--) {
                        if ((*x_boundaries)[j] != 0) {
                            prev_known = j;
                            break;
                        }
                    }
                    for (int j = i + 1; j < grid_size_lines; j++) {
                        if ((*x_boundaries)[j] != 0) {
                            next_known = j;
                            break;
                        }
                    }
                    if (prev_known >= 0 && next_known >= 0) {
                        float ratio = (float)(i - prev_known) / (next_known - prev_known);
                        (*x_boundaries)[i] = (*x_boundaries)[prev_known] +
                                           (int)(ratio * ((*x_boundaries)[next_known] - (*x_boundaries)[prev_known]));
                    }
                }
            }
        }
    } else {
        // No vertical lines detected, create evenly spaced boundaries
        for (int i = 0; i < grid_size_lines; i++) {
            (*x_boundaries)[i] = (i * grid_width) / (*num_cols);
        }
    }

    freeContours(filtered_horiz);
    freeContours(filtered_vert);

    printf("Final boundaries: %d rows (%d boundaries), %d cols (%d boundaries)\n",
           *num_rows, grid_size_lines, *num_cols, grid_size_lines);

    return 1;
}
