#include "analysis/grid_analysis.h"
#include "image/operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

int has_proper_grid_lines(const Image *grid_region)
{
    if (!grid_region || grid_region->width <= 0 || grid_region->height <= 0)
    {
        return 0;
    }

    Image horizontal_lines = {0};
    Image vertical_lines = {0};
    detect_grid_lines(grid_region, &horizontal_lines, &vertical_lines);

    if (horizontal_lines.width <= 0 || horizontal_lines.height <= 0 ||
        vertical_lines.width <= 0 || vertical_lines.height <= 0)
    {
        free_image(&horizontal_lines);
        free_image(&vertical_lines);
        return 0;
    }

    // Find contours in the detected lines
    Contours *horiz_contours = findContours(&horizontal_lines, 0);
    Contours *vert_contours = findContours(&vertical_lines, 0);

    free_image(&horizontal_lines);
    free_image(&vertical_lines);

    if (!horiz_contours || !vert_contours)
    {
        if (horiz_contours)
            freeContours(horiz_contours);
        if (vert_contours)
            freeContours(vert_contours);
        return 0;
    }

    // Filter contours by length - they should span at least 60% of the grid
    // width/height
    double min_horiz_length = grid_region->width * 0.6;
    double min_vert_length = grid_region->height * 0.6;

    Contours *filtered_horiz =
        filterContoursByLength(horiz_contours, min_horiz_length);
    Contours *filtered_vert =
        filterContoursByLength(vert_contours, min_vert_length);

    freeContours(horiz_contours);
    freeContours(vert_contours);

    if (!filtered_horiz || !filtered_vert)
    {
        if (filtered_horiz)
            freeContours(filtered_horiz);
        if (filtered_vert)
            freeContours(filtered_vert);
        return 0;
    }

    int horiz_line_count = filtered_horiz->count;
    int vert_line_count = filtered_vert->count;

    freeContours(filtered_horiz);
    freeContours(filtered_vert);

    printf(
        "Grid line detection: found %d horizontal lines, %d vertical lines\n",
        horiz_line_count, vert_line_count);

    if (horiz_line_count >= 3 && vert_line_count >= 3)
    {
        printf("Detected proper grid lines - will use line detection\n");
        return 1;
    }
    else
    {
        printf("No proper grid lines detected - will use letter-based "
               "detection\n");
        return 0;
    }
}

/**
 * Merge colinear contour segments that belong to the same line and extend short lines
 * to match the average length of other lines.
 * @param contours Input contours (will be modified/freed)
 * @param is_horizontal True for horizontal lines, false for vertical lines
 * @param grid_dimension Width for horizontal lines, height for vertical lines
 * @return New contours with merged and extended lines
 */
Contours *merge_colinear_segments_and_extend(Contours *contours, int is_horizontal,
                                           int grid_dimension)
{
    if (!contours || contours->count == 0)
        return NULL;

    const int COLINEAR_TOLERANCE = 5;
    const int GAP_TOLERANCE = 20;

    // Group contours by their axis position (Y for horizontal, X for vertical)
    typedef struct {
        Contour *contours;
        int count;
        int capacity;
        int axis_position;
    } LineGroup;

    LineGroup *groups = NULL;
    int group_count = 0;
    int group_capacity = 0;

    // Sort contours by bounding box position
    for (int i = 0; i < contours->count - 1; i++)
    {
        for (int j = 0; j < contours->count - i - 1; j++)
        {
            Rect rect1, rect2;
            boundingRect(&contours->contours[j], &rect1);
            boundingRect(&contours->contours[j + 1], &rect2);

            int pos1 = is_horizontal ? rect1.y : rect1.x;
            int pos2 = is_horizontal ? rect2.y : rect2.x;

            if (pos1 > pos2)
            {
                Contour temp = contours->contours[j];
                contours->contours[j] = contours->contours[j + 1];
                contours->contours[j + 1] = temp;
            }
        }
    }

    // Group colinear contours
    for (int i = 0; i < contours->count; i++)
    {
        Rect rect;
        boundingRect(&contours->contours[i], &rect);
        int axis_pos = is_horizontal ? rect.y : rect.x;

        // Find existing group or create new one
        int group_idx = -1;
        for (int g = 0; g < group_count; g++)
        {
            if (abs(groups[g].axis_position - axis_pos) <= COLINEAR_TOLERANCE)
            {
                group_idx = g;
                break;
            }
        }

        if (group_idx == -1)
        {
            if (group_count >= group_capacity)
            {
                group_capacity = group_capacity == 0 ? 8 : group_capacity * 2;
                groups = realloc(groups, sizeof(LineGroup) * group_capacity);
                if (!groups) return NULL;
            }

            groups[group_count].contours = NULL;
            groups[group_count].count = 0;
            groups[group_count].capacity = 0;
            groups[group_count].axis_position = axis_pos;
            group_idx = group_count++;
        }

        LineGroup *group = &groups[group_idx];
        if (group->count >= group->capacity)
        {
            group->capacity = group->capacity == 0 ? 4 : group->capacity * 2;
            group->contours = realloc(group->contours, sizeof(Contour) * group->capacity);
            if (!group->contours) return NULL;
        }
        group->contours[group->count++] = contours->contours[i];
    }

    free(contours->contours);
    free(contours);

    // Now merge segments within each group and create new contours
    Contours *result = (Contours *)malloc(sizeof(Contours));
    if (!result) {
        free(groups);
        return NULL;
    }

    result->contours = NULL;
    result->count = 0;
    result->capacity = group_count > 0 ? group_count : 1;

    result->contours = (Contour *)malloc(sizeof(Contour) * result->capacity);
    if (!result->contours) {
        free(result);
        free(groups);
        return NULL;
    }

    // Process each group
    for (int g = 0; g < group_count; g++)
    {
        LineGroup *group = &groups[g];
        if (group->count == 0) continue;

        // Sort contours in group by their start position
        for (int i = 0; i < group->count - 1; i++)
        {
            for (int j = 0; j < group->count - i - 1; j++)
            {
                Rect rect1, rect2;
                boundingRect(&group->contours[j], &rect1);
                boundingRect(&group->contours[j + 1], &rect2);

                int start1 = is_horizontal ? rect1.x : rect1.y;
                int start2 = is_horizontal ? rect2.x : rect2.y;

                if (start1 > start2)
                {
                    Contour temp = group->contours[j];
                    group->contours[j] = group->contours[j + 1];
                    group->contours[j + 1] = temp;
                }
            }
        }

        // Merge overlapping or adjacent segments
        Contour merged_contour;
        merged_contour.points = NULL;
        merged_contour.count = 0;
        merged_contour.capacity = 0;

        int min_pos = INT_MAX;
        int max_pos = INT_MIN;

        for (int i = 0; i < group->count; i++)
        {
            Rect rect;
            boundingRect(&group->contours[i], &rect);

            int start_pos = is_horizontal ? rect.x : rect.y;
            int end_pos = is_horizontal ? (rect.x + rect.width) : (rect.y + rect.height);

            if (i > 0)
            {
                int prev_end_pos = is_horizontal ?
                    (merged_contour.points[merged_contour.count - 1].x) :
                    (merged_contour.points[merged_contour.count - 1].y);

                if (abs(start_pos - prev_end_pos) <= GAP_TOLERANCE)
                {
                    if (is_horizontal)
                    {
                        // Extend horizontally
                        merged_contour.points[merged_contour.count - 1].x = end_pos;
                    }
                    else
                    {
                        // Extend vertically
                        merged_contour.points[merged_contour.count - 1].y = end_pos;
                    }
                    max_pos = end_pos > max_pos ? end_pos : max_pos;
                    continue;
                }
            }

            // Start new segment or add first point
            if (merged_contour.count == 0)
            {
                merged_contour.capacity = 4;
                merged_contour.points = (Point *)malloc(sizeof(Point) * merged_contour.capacity);
                if (!merged_contour.points) {
                    free(result->contours);
                    free(result);
                    free(groups);
                    return NULL;
                }

                merged_contour.points[0].x = is_horizontal ? start_pos : group->axis_position;
                merged_contour.points[0].y = is_horizontal ? group->axis_position : start_pos;
                merged_contour.count = 1;
            }

            if (merged_contour.count >= merged_contour.capacity)
            {
                merged_contour.capacity *= 2;
                merged_contour.points = realloc(merged_contour.points,
                    sizeof(Point) * merged_contour.capacity);
                if (!merged_contour.points) {
                    free(result->contours);
                    free(result);
                    free(groups);
                    return NULL;
                }
            }

            merged_contour.points[merged_contour.count].x = is_horizontal ? end_pos : group->axis_position;
            merged_contour.points[merged_contour.count].y = is_horizontal ? group->axis_position : end_pos;
            merged_contour.count++;

            min_pos = start_pos < min_pos ? start_pos : min_pos;
            max_pos = end_pos > max_pos ? end_pos : max_pos;
        }

        if (result->count >= result->capacity)
        {
            result->capacity *= 2;
            result->contours = realloc(result->contours, sizeof(Contour) * result->capacity);
            if (!result->contours) {
                free(result);
                free(groups);
                return NULL;
            }
        }

        result->contours[result->count++] = merged_contour;

        for (int i = 0; i < group->count; i++)
        {
            if (group->contours[i].points)
                free(group->contours[i].points);
        }
        free(group->contours);
    }

    // Calculate average length for extension
    double total_length = 0;
    int valid_lines = 0;

    for (int i = 0; i < result->count; i++)
    {
        double length = arcLength(&result->contours[i], 0);
        if (length > 0)
        {
            total_length += length;
            valid_lines++;
        }
    }

    double avg_length = valid_lines > 0 ? total_length / valid_lines : grid_dimension * 0.8;

    // Extend short lines to match average length
    for (int i = 0; i < result->count; i++)
    {
        Contour *contour = &result->contours[i];
        double current_length = arcLength(contour, 0);

        if (current_length < avg_length * 0.7)
        {
            // Extend the line to target length
            int target_length = (int)avg_length;

            if (contour->count >= 2)
            {
                // Get start and end points
                Point start = contour->points[0];
                Point end = contour->points[contour->count - 1];

                // Calculate direction vector
                int dx = end.x - start.x;
                int dy = end.y - start.y;

                // Calculate current length
                double current_len = sqrt(dx*dx + dy*dy);
                if (current_len > 0)
                {
                    double scale = target_length / current_len;

                    int center_x = (start.x + end.x) / 2;
                    int center_y = (start.y + end.y) / 2;

                    int new_start_x = center_x - (int)(dx * scale / 2);
                    int new_start_y = center_y - (int)(dy * scale / 2);
                    int new_end_x = center_x + (int)(dx * scale / 2);
                    int new_end_y = center_y + (int)(dy * scale / 2);

                    if (is_horizontal)
                    {
                        new_start_x = new_start_x < 0 ? 0 : new_start_x;
                        new_end_x = new_end_x > grid_dimension ? grid_dimension : new_end_x;
                        contour->points[0].x = new_start_x;
                        contour->points[contour->count - 1].x = new_end_x;
                    }
                    else
                    {
                        new_start_y = new_start_y < 0 ? 0 : new_start_y;
                        new_end_y = new_end_y > grid_dimension ? grid_dimension : new_end_y;
                        contour->points[0].y = new_start_y;
                        contour->points[contour->count - 1].y = new_end_y;
                    }
                }
            }
        }
    }

    free(groups);
    return result;

}

void detect_grid_lines(const Image *binary_grid, Image *horizontal_lines,
                       Image *vertical_lines)
{
    // Use adaptive kernels for different line thicknesses
    int min_dim = binary_grid->width < binary_grid->height
                      ? binary_grid->width
                      : binary_grid->height;
    int kernel_length =
        min_dim / 30;
    if (kernel_length < 5)
        kernel_length = 5; // Very small minimum for thin lines
    if (kernel_length > 40)
        kernel_length = 40; // Maximum size

    printf("Using morphological kernel length: %d\n", kernel_length);

    int thickness = 1;
    StructuringElement *horiz_kernel = getStructuringElement(
        0, kernel_length, thickness); // MORPH_RECT, length x thickness
    if (!horiz_kernel)
    {
        printf("Failed to create horizontal line kernel\n");
        return;
    }

    StructuringElement *vert_kernel = getStructuringElement(
        0, thickness, kernel_length); // MORPH_RECT, thickness x length
    if (!vert_kernel)
    {
        printf("Failed to create vertical line kernel\n");
        freeStructuringElement(horiz_kernel);
        return;
    }

    cpy_image(binary_grid, horizontal_lines);
    cpy_image(binary_grid, vertical_lines);

    // Detect horizontal lines: erode with horizontal kernel, then dilate
    morphologyEx(horizontal_lines, MORPH_ERODE, horiz_kernel, 1); // MORPH_ERODE
    morphologyEx(horizontal_lines, MORPH_DILATE, horiz_kernel,
                 1); // MORPH_DILATE

    // Detect vertical lines: erode with vertical kernel, then dilate
    morphologyEx(vertical_lines, MORPH_ERODE, vert_kernel, 1);  // MORPH_ERODE
    morphologyEx(vertical_lines, MORPH_DILATE, vert_kernel, 1); // MORPH_DILATE

    freeStructuringElement(horiz_kernel);
    freeStructuringElement(vert_kernel);
}

int extract_cell_boundaries_from_lines(const Image *horizontal_lines,
                                       const Image *vertical_lines,
                                       int grid_width, int grid_height,
                                       int **y_boundaries, int **x_boundaries,
                                       int *num_rows, int *num_cols)
{
    Contours *horiz_contours = findContours(horizontal_lines, 0);
    if (!horiz_contours)
    {
        printf("Failed to find horizontal line contours\n");
        return 0;
    }

    Contours *vert_contours = findContours(vertical_lines, 0);
    if (!vert_contours)
    {
        printf("Failed to find vertical line contours\n");
        freeContours(horiz_contours);
        return 0;
    }

    Contours *filtered_horiz =
        filterContoursByLength(horiz_contours, grid_width * 0.2);
    freeContours(horiz_contours);

    Contours *filtered_vert =
        filterContoursByLength(vert_contours, grid_height * 0.2);
    freeContours(vert_contours);

    Contours *processed_horiz = merge_colinear_segments_and_extend(filtered_horiz, 1, grid_width);
    Contours *processed_vert = merge_colinear_segments_and_extend(filtered_vert, 0, grid_height);

    if (!processed_horiz || !processed_vert)
    {
        if (processed_horiz) freeContours(processed_horiz);
        if (processed_vert) freeContours(processed_vert);
        return 0;
    }

    filtered_horiz = processed_horiz;
    filtered_vert = processed_vert;

    if (!filtered_horiz || !filtered_vert)
    {
        if (filtered_horiz)
            freeContours(filtered_horiz);
        if (filtered_vert)
            freeContours(filtered_vert);
        return 0;
    }

    int detected_horiz = filtered_horiz->count;
    int detected_vert = filtered_vert->count;

    if (detected_horiz < 3 || detected_vert < 3)
    {
        freeContours(filtered_horiz);
        freeContours(filtered_vert);
        return 0;
    }

    int grid_size_lines =
        detected_horiz < detected_vert ? detected_horiz : detected_vert;
    //! WARNING: This is a hardcoded value for the max grid size.
    if (grid_size_lines > 21)
        grid_size_lines = 21;

    *num_rows = grid_size_lines - 1;
    *num_cols = grid_size_lines - 1;

    *y_boundaries = (int *)malloc(sizeof(int) * grid_size_lines);
    *x_boundaries = (int *)malloc(sizeof(int) * grid_size_lines);
    if (!*y_boundaries || !*x_boundaries)
    {
        if (*y_boundaries)
            free(*y_boundaries);
        if (*x_boundaries)
            free(*x_boundaries);
        freeContours(filtered_horiz);
        freeContours(filtered_vert);
        return 0;
    }

    // Sort and extract Y boundaries from horizontal lines
    if (filtered_horiz->count > 0)
    {
        for (int i = 0; i < filtered_horiz->count - 1; i++)
        {
            for (int j = 0; j < filtered_horiz->count - i - 1; j++)
            {
                Rect rect1, rect2;
                boundingRect(&filtered_horiz->contours[j], &rect1);
                boundingRect(&filtered_horiz->contours[j + 1], &rect2);

                if (rect1.y > rect2.y)
                {
                    Contour temp = filtered_horiz->contours[j];
                    filtered_horiz->contours[j] =
                        filtered_horiz->contours[j + 1];
                    filtered_horiz->contours[j + 1] = temp;
                }
            }
        }

        // Extract Y boundaries - distribute evenly if we have fewer than
        // expected
        printf("Filtered horizontal count: %d\n", filtered_horiz->count);
        printf("Grid size lines: %d\n", grid_size_lines);
        if (filtered_horiz->count >= grid_size_lines)
        {
            for (int i = 0; i < grid_size_lines; i++)
            {
                Rect rect;
                boundingRect(&filtered_horiz->contours[i], &rect);
                (*y_boundaries)[i] = rect.y;
            }
        }
        else
        {
            // If we have fewer lines than expected, interpolate
            for (int i = 0; i < filtered_horiz->count; i++)
            {
                Rect rect;
                boundingRect(&filtered_horiz->contours[i], &rect);
                int target_idx = (int)((float)i / (filtered_horiz->count - 1) *
                                       (grid_size_lines - 1));
                if (target_idx < grid_size_lines)
                {
                    (*y_boundaries)[target_idx] = rect.y;
                }
            }
            // Fill in missing boundaries by interpolation
            for (int i = 0; i < grid_size_lines; i++)
            {
                if (i > 0 && (*y_boundaries)[i] == 0)
                {
                    // Interpolate between neighboring known boundaries
                    int prev_known = -1, next_known = -1;
                    for (int j = i - 1; j >= 0; j--)
                    {
                        if ((*y_boundaries)[j] != 0)
                        {
                            prev_known = j;
                            break;
                        }
                    }
                    for (int j = i + 1; j < grid_size_lines; j++)
                    {
                        if ((*y_boundaries)[j] != 0)
                        {
                            next_known = j;
                            break;
                        }
                    }
                    if (prev_known >= 0 && next_known >= 0)
                    {
                        float ratio =
                            (float)(i - prev_known) / (next_known - prev_known);
                        (*y_boundaries)[i] =
                            (*y_boundaries)[prev_known] +
                            (int)(ratio * ((*y_boundaries)[next_known] -
                                           (*y_boundaries)[prev_known]));
                    }
                }
            }
        }
    }
    else
    {
        // No horizontal lines detected, create evenly spaced boundaries
        for (int i = 0; i < grid_size_lines; i++)
        {
            (*y_boundaries)[i] = (i * grid_height) / (*num_rows);
        }
    }

    printf("Filtered vertical count: %d\n", filtered_vert->count);
    printf("Grid size lines: %d\n", grid_size_lines);
    // Sort and extract X boundaries from vertical lines
    if (filtered_vert->count > 0)
    {
        for (int i = 0; i < filtered_vert->count - 1; i++)
        {
            for (int j = 0; j < filtered_vert->count - i - 1; j++)
            {
                Rect rect1, rect2;
                boundingRect(&filtered_vert->contours[j], &rect1);
                boundingRect(&filtered_vert->contours[j + 1], &rect2);

                if (rect1.x > rect2.x)
                {
                    Contour temp = filtered_vert->contours[j];
                    filtered_vert->contours[j] = filtered_vert->contours[j + 1];
                    filtered_vert->contours[j + 1] = temp;
                }
            }
        }

        // Extract X boundaries - distribute evenly if we have fewer than
        // expected
        if (filtered_vert->count >= grid_size_lines)
        {
            // If we have enough lines, use the first grid_size_lines
            for (int i = 0; i < grid_size_lines; i++)
            {
                Rect rect;
                boundingRect(&filtered_vert->contours[i], &rect);
                (*x_boundaries)[i] = rect.x;
            }
        }
        else
        {
            // If we have fewer lines than expected, interpolate
            for (int i = 0; i < filtered_vert->count; i++)
            {
                Rect rect;
                boundingRect(&filtered_vert->contours[i], &rect);
                int target_idx = (int)((float)i / (filtered_vert->count - 1) *
                                       (grid_size_lines - 1));
                if (target_idx < grid_size_lines)
                {
                    (*x_boundaries)[target_idx] = rect.x;
                }
            }
            for (int i = 0; i < grid_size_lines; i++)
            {
                if (i > 0 && (*x_boundaries)[i] == 0)
                {
                    int prev_known = -1, next_known = -1;
                    for (int j = i - 1; j >= 0; j--)
                    {
                        if ((*x_boundaries)[j] != 0)
                        {
                            prev_known = j;
                            break;
                        }
                    }
                    for (int j = i + 1; j < grid_size_lines; j++)
                    {
                        if ((*x_boundaries)[j] != 0)
                        {
                            next_known = j;
                            break;
                        }
                    }
                    if (prev_known >= 0 && next_known >= 0)
                    {
                        float ratio =
                            (float)(i - prev_known) / (next_known - prev_known);
                        (*x_boundaries)[i] =
                            (*x_boundaries)[prev_known] +
                            (int)(ratio * ((*x_boundaries)[next_known] -
                                           (*x_boundaries)[prev_known]));
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < grid_size_lines; i++)
        {
            (*x_boundaries)[i] = (i * grid_width) / (*num_cols);
        }
    }

    freeContours(filtered_horiz);
    freeContours(filtered_vert);

    printf(
        "Final boundaries: %d rows (%d boundaries), %d cols (%d boundaries)\n",
        *num_rows, grid_size_lines, *num_cols, grid_size_lines);

    return 1;
}
