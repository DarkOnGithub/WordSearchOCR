#include "wordsearch/processor.h"
#include "analysis/contour_analysis.h"
#include "analysis/grid_analysis.h"
#include "image/image.h"
#include "processing/preprocessing.h"
#include "wordsearch/word_detection.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <math.h>

int process_wordsearch_image(const char *image_path,
                             CreateButtonCallback create_button_callback, int *num_rows, int *num_cols,
                             Rect **cell_bounding_boxes, int *num_cells,
                             int *crop_offset_x, int *crop_offset_y)
{
    if (!image_path || !num_rows || !num_cols || !cell_bounding_boxes || !num_cells ||
        !crop_offset_x || !crop_offset_y)
    {
        fprintf(stderr, "Error: Invalid parameters provided\n");
        return 1;
    }

    // Initialize output parameters
    *num_rows = 0;
    *num_cols = 0;
    *cell_bounding_boxes = NULL;
    *num_cells = 0;
    *crop_offset_x = 0;
    *crop_offset_y = 0;

    printf("Processing word search image: %s\n", image_path);

    Image image;
    if (!load_and_preprocess_image(image_path, &image, create_button_callback))
    {
        fprintf(stderr, "Failed to load and preprocess image\n");
        return 1;
    }

    Image original_image;
    load_image(image_path, &original_image);

    Image grid_image;
    Rect grid_bounds;
    if (!extract_grid_region(&image, &original_image, &grid_image, &grid_bounds,
                             create_button_callback))
    {
        fprintf(stderr, "Failed to extract grid region\n");
        free_image(&image);
        free_image(&original_image);
        return 1;
    }

    if (!process_grid_for_ocr(&grid_image, create_button_callback))
    {
        fprintf(stderr, "Failed to process grid for OCR\n");
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    Contours *letters_contours = findContours(&grid_image, 0);
    if (!letters_contours)
    {
        fprintf(stderr, "Failed to find contours in grid image\n");
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    Contours *valid_letters = filter_letter_contours(letters_contours);
    if (!valid_letters)
    {
        fprintf(stderr, "Failed to filter letter contours\n");
        freeContours(letters_contours);
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    int use_line_detection = has_proper_grid_lines(&grid_image);

    int local_num_rows = 0, local_num_cols = 0;
    int *y_boundaries = NULL;
    int *x_boundaries = NULL;

    if (use_line_detection)
    {
        printf("Using line-based detection for grid with inner contours\n");

        Image horizontal_lines = {0};
        Image vertical_lines = {0};
        detect_grid_lines(&grid_image, &horizontal_lines, &vertical_lines);

        save_image("step_09_horizontal_lines.png", &horizontal_lines);
        save_image("step_09_vertical_lines.png", &vertical_lines);

        if (create_button_callback)
        {
            create_button_callback("Horizontal Lines",
                                   "step_09_horizontal_lines.png");
            create_button_callback("Vertical Lines",
                                   "step_09_vertical_lines.png");
        }

        if (!extract_cell_boundaries_from_lines(
                &horizontal_lines, &vertical_lines, grid_image.width,
                grid_image.height, &y_boundaries, &x_boundaries, &local_num_rows,
                &local_num_cols))
        {
            printf("Failed to extract cell boundaries from lines, falling back "
                   "to letter-based detection\n");
            use_line_detection = 0;
            free_image(&horizontal_lines);
            free_image(&vertical_lines);
        }
        else
        {
            free_image(&horizontal_lines);
            free_image(&vertical_lines);
        }
    }

    Image text_region = {0};
    if (!use_line_detection)
    {
        printf(
            "Using letter-based detection for grid without inner contours\n");

        if (!determine_text_region(
                &original_image, grid_bounds.x, grid_bounds.y,
                grid_bounds.width, grid_bounds.height, valid_letters,
                &text_region, crop_offset_x, crop_offset_y))
        {
            fprintf(stderr, "Failed to determine text region\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            return 1;
        }

        if (determine_grid_dimensions_from_letters(valid_letters, *crop_offset_x,
                                                   *crop_offset_y, &local_num_rows,
                                                   &local_num_cols) != 0)
        {
            fprintf(stderr, "Failed to determine grid dimensions\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            free_image(&text_region);
            return 1;
        }

        if (generate_cell_boundaries_from_letters(valid_letters, local_num_rows, local_num_cols,
                                                  &y_boundaries, &x_boundaries) != 0)
        {
            fprintf(stderr, "Failed to generate cell boundaries\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            free_image(&text_region);
            return 1;
        }
    }
    else
    {
        // Extract text region from original image like letter-based detection
        if (!determine_text_region(&original_image, grid_bounds.x, grid_bounds.y,
                                   grid_bounds.width, grid_bounds.height, NULL,
                                   &text_region, crop_offset_x, crop_offset_y))
        {
            fprintf(stderr, "Failed to determine text region in line detection\n");
            return 1;
        }
    }

    save_image("step_08_text_region.png", &text_region);

    if (create_button_callback)
    {
        create_button_callback("Text Region", "step_08_text_region.png");
    }

    Image debug_grid = {0};
    cpy_image(&text_region, &debug_grid);
    gray_to_rgba(&debug_grid);
    for (int i = 0; i <= local_num_rows; i++)
    {
        int y = y_boundaries[i];
        draw_rectangle(&debug_grid, 0, y, text_region.width, 1, true, 3,
                       0xFF0000FF);
    }

    for (int i = 0; i <= local_num_cols; i++)
    {
        int x = x_boundaries[i];
        draw_rectangle(&debug_grid, x, 0, 1, text_region.height, true, 3,
                       0xFF0000FF);
    }

    save_image("step_9_grid_region.png", &debug_grid);

    if (create_button_callback)
    {
        create_button_callback("Grid Region", "step_9_grid_region.png");
    }

    free_image(&debug_grid);

    if (extract_cell_images(&text_region, y_boundaries, x_boundaries, local_num_rows,
                            local_num_cols) != 0)
    {
        fprintf(stderr, "Failed to extract cell images\n");
    }

    if (create_reconstructed_grid(local_num_rows, local_num_cols, create_button_callback) !=
        0)
    {
        fprintf(stderr, "Failed to create reconstructed grid\n");
    }

    printf("Extracted grid: %d rows x %d columns\n", local_num_rows, local_num_cols);

    // Create cell bounding boxes array
    int total_cells = local_num_rows * local_num_cols;
    *cell_bounding_boxes = (Rect *)malloc(sizeof(Rect) * total_cells);
    if (!*cell_bounding_boxes)
    {
        fprintf(stderr, "Failed to allocate memory for cell bounding boxes\n");
        free(y_boundaries);
        free(x_boundaries);
        freeContours(letters_contours);
        freeContours(valid_letters);
        free_image(&text_region);
        free_image(&grid_image);
        free_image(&original_image);
        free_image(&image);
        return 1;
    }

    // Fill in the bounding boxes for each cell (relative to text region)
    for (int row = 0; row < local_num_rows; row++)
    {
        for (int col = 0; col < local_num_cols; col++)
        {
            int cell_index = row * local_num_cols + col;
            (*cell_bounding_boxes)[cell_index].x = x_boundaries[col];
            (*cell_bounding_boxes)[cell_index].y = y_boundaries[row];
            (*cell_bounding_boxes)[cell_index].width = x_boundaries[col + 1] - x_boundaries[col];
            (*cell_bounding_boxes)[cell_index].height = y_boundaries[row + 1] - y_boundaries[row];
        }
    }

    // Set output parameters
    *num_rows = local_num_rows;
    *num_cols = local_num_cols;
    *num_cells = total_cells;
    // Add grid bounds to get absolute offset in original image
    *crop_offset_x += grid_bounds.x;
    *crop_offset_y += grid_bounds.y;
    // Free internal boundary arrays (cell bounding boxes are returned to caller)
    free(y_boundaries);
    free(x_boundaries);
    freeContours(letters_contours);
    freeContours(valid_letters);
    free_image(&text_region);
    free_image(&grid_image);
    free_image(&original_image);
    free_image(&image);

    return 0;
}

void clear_directory(const char *dirname)
{
    DIR *dir = opendir(dirname);
    if (!dir)
    {
        printf("Could not open %s directory for clearing\n", dirname);
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL)
    {
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0 ||
            strcmp(entry->d_name, ".gitkeep") == 0)
        {
            continue;
        }

        char filepath[512];
        snprintf(filepath, sizeof(filepath), "%s/%s", dirname, entry->d_name);
        if (unlink(filepath) == 0)
        {
            // printf("Removed: %s\n", filepath);
        }
        else
        {
            printf("Failed to remove: %s\n", filepath);
        }
    }

    closedir(dir);
    printf("%s directory cleared (except .gitkeep)\n", dirname);
}

int extract_cell_images(const Image *grid_region, const int *y_boundaries,
                        const int *x_boundaries, int num_rows, int num_cols)
{
    if (!grid_region || !y_boundaries || !x_boundaries)
    {
        return 1;
    }

    clear_directory("cells");

    printf("Extracting individual cells for OCR processing...\n");
    for (int row = 0; row < num_rows; row++)
    {
        for (int col = 0; col < num_cols; col++)
        {
            int y1 = y_boundaries[row];
            int y2 = y_boundaries[row + 1];
            int x1 = x_boundaries[col];
            int x2 = x_boundaries[col + 1];
            int cell_width_px = x2 - x1;
            int cell_height_px = y2 - y1;

            if (cell_width_px > 0 && cell_height_px > 0)
            {
                Image cell_roi = {0};
                extract_rectangle(grid_region, x1, y1, cell_width_px,
                                  cell_height_px, &cell_roi);
                char filename[256];
                sprintf(filename, "cells/cell_%d_%d.png", row, col);
                save_image(filename, &cell_roi);

                free_image(&cell_roi);
            }
        }
    }

    return 0;
}

int create_reconstructed_grid(int num_rows, int num_cols,
                              CreateButtonCallback create_button_callback)
{
    if (num_rows <= 0 || num_cols <= 0)
    {
        return 1;
    }

    // Load the first available cell to get dimensions
    Image first_cell = {0};
    char first_filename[256];
    int found_cell = 0;

    // Try different cells until we find one that exists
    for (int r = 0; r < num_rows && !found_cell; r++)
    {
        for (int c = 0; c < num_cols && !found_cell; c++)
        {
            sprintf(first_filename, "cells/cell_%d_%d.png", r, c);
            load_image(first_filename, &first_cell);
            if (first_cell.rgba_pixels || first_cell.gray_pixels)
            {
                found_cell = 1;
            }
        }
    }

    if (!found_cell)
    {
        fprintf(stderr, "Failed to load any cell image for dimensions\n");
        return 1;
    }

    int cell_width = first_cell.width;
    int cell_height = first_cell.height;
    free_image(&first_cell);

    int spacing = 5;
    int grid_width = num_cols * cell_width + (num_cols - 1) * spacing;
    int grid_height = num_rows * cell_height + (num_rows - 1) * spacing;

    Image reconstructed_grid = {0};
    reconstructed_grid.width = grid_width;
    reconstructed_grid.height = grid_height;
    reconstructed_grid.is_grayscale = false;

    size_t pixel_count = grid_width * grid_height;
    reconstructed_grid.rgba_pixels =
        (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
    if (!reconstructed_grid.rgba_pixels)
    {
        fprintf(stderr,
                "Failed to allocate memory for reconstructed grid image\n");
        return 1;
    }

    for (size_t i = 0; i < pixel_count; i++)
    {
        reconstructed_grid.rgba_pixels[i] =
            0xFF0000FF;
    }

    for (int row = 0; row < num_rows; row++)
    {
        for (int col = 0; col < num_cols; col++)
        {
            char filename[256];
            sprintf(filename, "cells/cell_%d_%d.png", row, col);

            Image cell = {0};
            load_image(filename, &cell);
            if (!cell.rgba_pixels && !cell.gray_pixels)
            {
                fprintf(stderr, "Failed to load cell image: %s\n", filename);
                continue;
            }

            int dest_x = col * (cell_width + spacing);
            int dest_y = row * (cell_height + spacing);

            for (int y = 0; y < cell_height && (dest_y + y) < grid_height; y++)
            {
                for (int x = 0; x < cell_width && (dest_x + x) < grid_width;
                     x++)
                {
                    int dest_idx = (dest_y + y) * grid_width + (dest_x + x);

                    if (cell.is_grayscale)
                    {
                        uint8_t gray = cell.gray_pixels[y * cell.width + x];
                        reconstructed_grid.rgba_pixels[dest_idx] =
                            (gray << 24) | (gray << 16) | (gray << 8) | gray;
                    }
                    else
                    {
                        reconstructed_grid.rgba_pixels[dest_idx] =
                            cell.rgba_pixels[y * cell.width + x];
                    }
                }
            }

            free_image(&cell);
        }
    }

    save_image("step_10_reconstructed_grid.png", &reconstructed_grid);

    if (create_button_callback)
    {
        create_button_callback("Reconstructed Grid",
                               "step_10_reconstructed_grid.png");
    }

    free_image(&reconstructed_grid);
    return 0;
}

int determine_grid_dimensions_from_letters(Contours *valid_letters,
                                           int crop_offset_x, int crop_offset_y,
                                           int *num_rows, int *num_cols)
{
    if (!valid_letters || !num_rows || !num_cols)
    {
        return 1;
    }

    translateContours(valid_letters, -crop_offset_x, -crop_offset_y);

#define ROW_TOLERANCE 10

    LetterRow *rows = NULL;
    int rows_capacity = 0;
    int actual_num_rows = 0;

    for (int i = 0; i < valid_letters->count; i++)
    {
        Rect rect;
        if (!boundingRect(&valid_letters->contours[i], &rect))
        {
            continue;
        }

        int letter_y = rect.y;
        int found_row_idx = -1;

        for (int j = 0; j < actual_num_rows; j++)
        {
            if (abs(letter_y - rows[j].row_y) <= ROW_TOLERANCE)
            {
                found_row_idx = j;
                break;
            }
        }

        if (found_row_idx == -1)
        {
            if (actual_num_rows >= rows_capacity)
            {
                rows_capacity = rows_capacity == 0 ? 4 : rows_capacity * 2;
                rows = (LetterRow *)realloc(rows,
                                            sizeof(LetterRow) * rows_capacity);
                if (!rows)
                {
                    fprintf(stderr, "Failed to allocate memory for rows\n");
                    return 1;
                }
            }

            found_row_idx = actual_num_rows;
            rows[actual_num_rows].row_y = letter_y;
            rows[actual_num_rows].letters = NULL;
            rows[actual_num_rows].count = 0;
            rows[actual_num_rows].capacity = 0;
            actual_num_rows++;
        }

        LetterRow *row = &rows[found_row_idx];
        if (row->count >= row->capacity)
        {
            row->capacity = row->capacity == 0 ? 4 : row->capacity * 2;
            row->letters =
                (Rect *)realloc(row->letters, sizeof(Rect) * row->capacity);
            if (!row->letters)
            {
                fprintf(stderr,
                        "Failed to allocate memory for letters in row\n");
                for (int k = 0; k < actual_num_rows; k++)
                {
                    free(rows[k].letters);
                }
                free(rows);
                return 1;
            }
        }

        row->letters[row->count++] = rect;
    }

    *num_cols = 0;
    for (int i = 0; i < actual_num_rows; i++)
    {
        if (rows[i].count > *num_cols)
        {
            *num_cols = rows[i].count;
        }
    }

    *num_rows = actual_num_rows;

    printf("Detected grid structure: %d rows x %d columns\n", *num_rows,
           *num_cols);

    for (int i = 0; i < actual_num_rows; i++)
    {
        free(rows[i].letters);
    }
    free(rows);

    return 0;
}

int generate_safe_cell_boundaries(const Image *grid_region, int num_rows,
                                  int num_cols, int **y_boundaries,
                                  int **x_boundaries)
{
    if (!grid_region || !y_boundaries || !x_boundaries || num_rows <= 0 ||
        num_cols <= 0)
    {
        return 1;
    }

    int cell_width = grid_region->width / num_cols;
    int cell_height = grid_region->height / num_rows;
    printf("Cell dimensions: %dx%d\n", cell_width, cell_height);

    Image binary_debug = {0};
    cpy_image(grid_region, &binary_debug);
    convert_to_grayscale(&binary_debug);

    double debug_otsu_threshold = threshold(&binary_debug, 255);
    if (debug_otsu_threshold < 0)
    {
        fprintf(stderr, "Failed to apply Otsu's thresholding for debug\n");
        free_image(&binary_debug);
        return 1;
    }

    double mean_val = 0.0;
    for (int i = 0; i < binary_debug.width * binary_debug.height; i++)
    {
        mean_val += binary_debug.gray_pixels[i];
    }
    mean_val /= (binary_debug.width * binary_debug.height);

    if (mean_val > 127)
    {
        for (int i = 0; i < binary_debug.width * binary_debug.height; i++)
        {
            binary_debug.gray_pixels[i] = 255 - binary_debug.gray_pixels[i];
        }
    }

    *y_boundaries = (int *)malloc(sizeof(int) * (num_rows + 1));
    *x_boundaries = (int *)malloc(sizeof(int) * (num_cols + 1));
    if (!*y_boundaries || !*x_boundaries)
    {
        fprintf(stderr, "Failed to allocate memory for boundary arrays\n");
        free_image(&binary_debug);
        if (*y_boundaries)
            free(*y_boundaries);
        if (*x_boundaries)
            free(*x_boundaries);
        return 1;
    }

    for (int row = 0; row <= num_rows; row++)
    {
        int y = row * cell_height;
        int safe_y = find_safe_line_position(
            y, 1, 4, &binary_debug, grid_region->width, grid_region->height);
        (*y_boundaries)[row] = safe_y;
    }

    for (int col = 0; col <= num_cols; col++)
    {
        int x = col * cell_width;
        int safe_x = find_safe_line_position(
            x, 0, 4, &binary_debug, grid_region->width, grid_region->height);
        (*x_boundaries)[col] = safe_x;
    }

    free_image(&binary_debug);
    return 0;
}

int generate_cell_boundaries_from_letters(Contours *valid_letters, int num_rows,
                                          int num_cols, int **y_boundaries,
                                          int **x_boundaries)
{
    if (!valid_letters || !y_boundaries || !x_boundaries || num_rows <= 0 ||
        num_cols <= 0)
    {
        return 1;
    }

    *y_boundaries = (int *)malloc(sizeof(int) * (num_rows + 1));
    *x_boundaries = (int *)malloc(sizeof(int) * (num_cols + 1));
    if (!*y_boundaries || !*x_boundaries)
    {
        fprintf(stderr, "Failed to allocate memory for boundary arrays\n");
        if (*y_boundaries)
            free(*y_boundaries);
        if (*x_boundaries)
            free(*x_boundaries);
        return 1;
    }

#define ROW_TOLERANCE 10

    LetterRow *rows = NULL;
    int rows_capacity = 0;
    int actual_num_rows = 0;

    // First pass: group letters by rows
    for (int i = 0; i < valid_letters->count; i++)
    {
        Rect rect;
        if (!boundingRect(&valid_letters->contours[i], &rect))
        {
            continue;
        }

        int letter_y = rect.y + rect.height / 2;
        int found_row_idx = -1;

        for (int j = 0; j < actual_num_rows; j++)
        {
            if (abs(letter_y - rows[j].row_y) <= ROW_TOLERANCE)
            {
                found_row_idx = j;
                break;
            }
        }

        if (found_row_idx == -1)
        {
            if (actual_num_rows >= rows_capacity)
            {
                rows_capacity = rows_capacity == 0 ? 4 : rows_capacity * 2;
                rows = (LetterRow *)realloc(rows,
                                            sizeof(LetterRow) * rows_capacity);
                if (!rows)
                {
                    fprintf(stderr, "Failed to allocate memory for rows\n");
                    free(*y_boundaries);
                    free(*x_boundaries);
                    return 1;
                }
            }

            found_row_idx = actual_num_rows;
            rows[actual_num_rows].row_y = letter_y;
            rows[actual_num_rows].letters = NULL;
            rows[actual_num_rows].count = 0;
            rows[actual_num_rows].capacity = 0;
            actual_num_rows++;
        }

        LetterRow *row = &rows[found_row_idx];
        if (row->count >= row->capacity)
        {
            row->capacity = row->capacity == 0 ? 4 : row->capacity * 2;
            row->letters =
                (Rect *)realloc(row->letters, sizeof(Rect) * row->capacity);
            if (!row->letters)
            {
                fprintf(stderr,
                        "Failed to allocate memory for letters in row\n");
                for (int k = 0; k < actual_num_rows; k++)
                {
                    free(rows[k].letters);
                }
                free(rows);
                free(*y_boundaries);
                free(*x_boundaries);
                return 1;
            }
        }

        row->letters[row->count++] = rect;
    }

    for (int i = 0; i < actual_num_rows - 1; i++)
    {
        for (int j = i + 1; j < actual_num_rows; j++)
        {
            if (rows[i].row_y > rows[j].row_y)
            {
                LetterRow temp = rows[i];
                rows[i] = rows[j];
                rows[j] = temp;
            }
        }
    }

    if (actual_num_rows > 0)
    {
        // Find minimum and maximum Y positions across all letters
        int min_y = INT_MAX;
        int max_y = INT_MIN;

        for (int i = 0; i < valid_letters->count; i++)
        {
            Rect rect;
            if (boundingRect(&valid_letters->contours[i], &rect))
            {
                if (rect.y < min_y) min_y = rect.y;
                if (rect.y + rect.height > max_y) max_y = rect.y + rect.height;
            }
        }

        // Set top boundary
        (*y_boundaries)[0] = min_y;

        // For multiple rows, place boundaries at centers of gaps between rows
        if (actual_num_rows > 1)
        {
            for (int i = 0; i < actual_num_rows - 1; i++)
            {
                int current_row_bottom = rows[i].row_y;
                int next_row_top = rows[i + 1].row_y;

                // Find the actual bottom of letters in current row
                for (int j = 0; j < rows[i].count; j++)
                {
                    if (rows[i].letters[j].y + rows[i].letters[j].height > current_row_bottom)
                    {
                        current_row_bottom = rows[i].letters[j].y + rows[i].letters[j].height;
                    }
                }

                // Find the actual top of letters in next row
                for (int j = 0; j < rows[i + 1].count; j++)
                {
                    if (rows[i + 1].letters[j].y < next_row_top)
                    {
                        next_row_top = rows[i + 1].letters[j].y;
                    }
                }

                // Place boundary at center of gap
                (*y_boundaries)[i + 1] = (current_row_bottom + next_row_top) / 2;
            }
        }

        // Set bottom boundary
        (*y_boundaries)[num_rows] = max_y;
    }
    else
    {
        // Fallback if no letters found
        (*y_boundaries)[0] = 0;
        for (int i = 1; i <= num_rows; i++)
        {
            (*y_boundaries)[i] = i * 50; //because why not
        }
    }

    // Generate X boundaries based on column positions within each row
    if (actual_num_rows > 0)
    {
        // Use the row with the most letters to determine column positions
        int max_letters_row = 0;
        for (int i = 1; i < actual_num_rows; i++)
        {
            if (rows[i].count > rows[max_letters_row].count)
            {
                max_letters_row = i;
            }
        }

        LetterRow *reference_row = &rows[max_letters_row];

        // Sort letters in the reference row by X position
        for (int i = 0; i < reference_row->count - 1; i++)
        {
            for (int j = i + 1; j < reference_row->count; j++)
            {
                if (reference_row->letters[i].x > reference_row->letters[j].x)
                {
                    Rect temp = reference_row->letters[i];
                    reference_row->letters[i] = reference_row->letters[j];
                    reference_row->letters[j] = temp;
                }
            }
        }

        // Find overall min and max X positions
        int min_x = INT_MAX;
        int max_x = INT_MIN;

        for (int i = 0; i < valid_letters->count; i++)
        {
            Rect rect;
            if (boundingRect(&valid_letters->contours[i], &rect))
            {
                if (rect.x < min_x) min_x = rect.x;
                if (rect.x + rect.width > max_x) max_x = rect.x + rect.width;
            }
        }

        (*x_boundaries)[0] = min_x;

        // For multiple columns, place boundaries at centers of gaps between letters
        if (reference_row->count > 1)
        {
            for (int i = 0; i < reference_row->count - 1 && i < num_cols - 1; i++)
            {
                int current_letter_right = reference_row->letters[i].x + reference_row->letters[i].width;
                int next_letter_left = reference_row->letters[i + 1].x;

                // Place boundary at center of gap
                (*x_boundaries)[i + 1] = (current_letter_right + next_letter_left) / 2;
            }
        }

        // Fill remaining boundaries if needed
        for (int i = reference_row->count; i < num_cols; i++)
        {
            (*x_boundaries)[i] = (*x_boundaries)[i-1] + 50; //still because why not
        }

        // Set right boundary
        (*x_boundaries)[num_cols] = max_x;
    }
    else
    {
        // Fallback if no letters found
        (*x_boundaries)[0] = 0;
        for (int i = 1; i <= num_cols; i++)
        {
            (*x_boundaries)[i] = i * 50; //still still because why not :)
        }
    }

    for (int i = 0; i < actual_num_rows; i++)
    {
        free(rows[i].letters);
    }
    free(rows);

    return 0;
}

int process_word_detection(const char *image_path,
                           CreateButtonCallback create_button_callback)
{
    if (!image_path)
    {
        fprintf(stderr, "Error: No image path provided\n");
        return 1;
    }

    printf("Processing word detection: %s\n", image_path);

    char debug_prefix[256] = "word_detection";

    BoundingBoxArray *detected_words = detect_words(image_path, debug_prefix);

    if (!detected_words)
    {
        fprintf(stderr, "Failed to detect words\n");
        return 1;
    }

    if (create_button_callback)
    {
        create_button_callback("Original Image",
                               "word_detection_01_original.png");
        create_button_callback("Gaussian Blur",
                               "word_detection_02_blurred.png");
        create_button_callback("Otsu Threshold",
                               "word_detection_03_otsu_threshold.png");
        create_button_callback("Mean Threshold",
                               "word_detection_05_mean_threshold.png");
        create_button_callback("Combined Threshold",
                               "word_detection_06_combined_threshold.png");
        create_button_callback("Morphology Close",
                               "word_detection_07_morphology_closed.png");
        create_button_callback("Dilated", "word_detection_08_dilated.png");
        create_button_callback("Eroded", "word_detection_09_eroded.png");
    }

    Image original_image;
    load_image(image_path, &original_image);

    if (original_image.rgba_pixels || original_image.gray_pixels)
    {
        Image word_detection_result;
        cpy_image(&original_image, &word_detection_result);

        draw_bounding_boxes(&word_detection_result, detected_words, 0xFF00FF00,
                            3);

        save_image("word_detection_10_detected_words.png",
                   &word_detection_result);

        if (create_button_callback)
        {
            create_button_callback("Detected Words",
                                   "word_detection_10_detected_words.png");
        }

        free_image(&word_detection_result);
    }

    // Clear the words directory before saving new images
    clear_directory("words");

    // Extract and save individual word images from the main group
    for (int i = 0; i < detected_words->count; i++)
    {
        BoundingBox box = detected_words->boxes[i];

        // Extract the word region from the original image
        Image word_image = {0};
        extract_rectangle(&original_image, box.x, box.y, box.width, box.height, &word_image);
        char word_filename[256];
        snprintf(word_filename, sizeof(word_filename), "words/word_%d.png", i + 1);
        save_image(word_filename, &word_image);

        free_image(&word_image);
    }

    printf("Saved %d word images to words/ folder\n", detected_words->count);

    free_image(&original_image);
    freeBoundingBoxArray(detected_words);
    free(detected_words);

    printf("Word detection processing completed\n");
    return 0;
}

int draw_solved_words(const char *image_path, WordMatch **word_matches, int num_matches,
                      int num_rows, int num_cols, const Rect *cell_bounding_boxes,
                      int text_region_offset_x, int text_region_offset_y, const char *output_path)
{
    if (!image_path || !word_matches || !cell_bounding_boxes || !output_path || num_matches <= 0)
    {
        fprintf(stderr, "Invalid parameters for draw_solved_words\n");
        return 1;
    }

    printf("Drawing solved words on image: %s\n", image_path);

    // Load the original image
    Image original_image;
    load_image(image_path, &original_image);
    if ((!original_image.is_grayscale && !original_image.rgba_pixels) ||
        (original_image.is_grayscale && !original_image.gray_pixels))
    {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        return 1;
    }

    // Convert to RGBA if needed for drawing
    if (original_image.is_grayscale)
    {
        gray_to_rgba(&original_image);
    }

    printf("Using text region offset: (%d, %d)\n", text_region_offset_x, text_region_offset_y);

    // Define semi-transparent colors for the capsules (30 colors)
    uint32_t colors[] = {
        0x40FF0000, // Red
        0x4000FF00, // Green
        0x400000FF, // Blue
        0x40FFFF00, // Yellow
        0x40FF00FF, // Magenta
        0x4000FFFF, // Cyan
        0x40FFA500, // Orange
        0x40800080, // Purple
        0x40008080, // Teal
        0x40808000, // Olive
        0x40FF4500, // Orange Red
        0x4000FF7F, // Spring Green
        0x404B0082, // Indigo
        0x40FF1493, // Deep Pink
        0x4000CED1, // Dark Turquoise
        0x40FF6347, // Tomato
        0x4032CD32, // Lime Green
        0x408B008B, // Dark Magenta
        0x4040E0D0, // Turquoise
        0x40FF8C00, // Dark Orange
        0x409ACD32, // Yellow Green
        0x40800000, // Dark Red
        0x40000080, // Navy
        0x40FFD700, // Gold
        0x40DA70D6, // Orchid
        0x4020B2AA, // Light Sea Green
        0x40DC143C, // Crimson
        0x40008000, // Dark Green
        0x408B4513, // Saddle Brown
        0x406B8E23  // Olive Drab
    };
    int num_colors = sizeof(colors) / sizeof(colors[0]);

    // Draw each solved word
    for (int i = 0; i < num_matches; i++)
    {
        WordMatch *match = word_matches[i];
        if (!match)
            continue;

        // Calculate word length
        int word_length = strlen(match->word_str);

        // Calculate end position based on direction
        int end_row = match->start_pos.row;
        int end_col = match->start_pos.col;

        // Direction deltas and angles
        double angle = 0.0; // angle in degrees
        if (strcmp(match->direction, "Right") == 0) {
            end_col += word_length - 1;
            angle = 0.0;
        } else if (strcmp(match->direction, "Left") == 0) {
            end_col -= word_length - 1;
            angle = 180.0;
        } else if (strcmp(match->direction, "Down") == 0) {
            end_row += word_length - 1;
            angle = 90.0;
        } else if (strcmp(match->direction, "Up") == 0) {
            end_row -= word_length - 1;
            angle = 270.0;
        } else if (strcmp(match->direction, "DownRight") == 0) {
            end_row += word_length - 1;
            end_col += word_length - 1;
            angle = 45.0;
        } else if (strcmp(match->direction, "DownLeft") == 0) {
            end_row += word_length - 1;
            end_col -= word_length - 1;
            angle = 135.0;
        } else if (strcmp(match->direction, "UpRight") == 0) {
            end_row -= word_length - 1;
            end_col += word_length - 1;
            angle = 315.0;
        } else if (strcmp(match->direction, "UpLeft") == 0) {
            end_row -= word_length - 1;
            end_col -= word_length - 1;
            angle = 225.0;
        }

        // Get bounding boxes for start and end cells
        if (match->start_pos.row < 0 || match->start_pos.row >= num_rows ||
            match->start_pos.col < 0 || match->start_pos.col >= num_cols ||
            end_row < 0 || end_row >= num_rows ||
            end_col < 0 || end_col >= num_cols)
        {
            fprintf(stderr, "Word position out of bounds for word: %s\n", match->word_str);
            continue;
        }

        Rect start_cell = cell_bounding_boxes[match->start_pos.row * num_cols + match->start_pos.col];
        Rect end_cell = cell_bounding_boxes[end_row * num_cols + end_col];

        // Apply offset to get coordinates in original image
        start_cell.x += text_region_offset_x;
        start_cell.y += text_region_offset_y;
        end_cell.x += text_region_offset_x;
        end_cell.y += text_region_offset_y;

        // Calculate the center points of start and end cells
        int start_center_x = start_cell.x + start_cell.width / 2;
        int start_center_y = start_cell.y + start_cell.height / 2;
        int end_center_x = end_cell.x + end_cell.width / 2;
        int end_center_y = end_cell.y + end_cell.height / 2;

        // Calculate the length and width of the capsule
        double dx = end_center_x - start_center_x;
        double dy = end_center_y - start_center_y;
        double distance = sqrt(dx * dx + dy * dy);

        // Capsule dimensions: smaller overall size
        int capsule_length = (int)(distance + start_cell.width * 0.8); // Smaller circular ends
        int capsule_width = start_cell.height + 4; // Narrower body

        // Calculate capsule center
        int capsule_center_x = (start_center_x + end_center_x) / 2;
        int capsule_center_y = (start_center_y + end_center_y) / 2;

        // Choose color based on word index
        uint32_t color = colors[i % num_colors];

        // Draw the angled capsule with appropriately sized rounded ends
        int capsule_radius = capsule_width / 2; // Standard radius for circular ends
        draw_angled_capsule(&original_image, capsule_center_x, capsule_center_y,
                           capsule_length, capsule_width, angle, capsule_radius, color);

        printf("Drew angled capsule for word '%s' at (%d,%d) to (%d,%d), direction: %s, angle: %.1f\n",
               match->word_str, match->start_pos.row, match->start_pos.col, end_row, end_col,
               match->direction, angle);
    }

    // Save the annotated image
    save_image(output_path, &original_image);

    printf("Saved annotated image with %d solved words to: %s\n", num_matches, output_path);

    // Cleanup
    free_image(&original_image);

    return 0;
}