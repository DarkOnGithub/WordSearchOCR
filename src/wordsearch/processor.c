#include "../../include/wordsearch/processor.h"
#include "../../include/analysis/contour_analysis.h"
#include "../../include/analysis/grid_analysis.h"
#include "../../include/processing/preprocessing.h"
#include "../../include/wordsearch/word_detection.h"
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>

int process_wordsearch_image(const char *image_path,
                             CreateButtonCallback create_button_callback)
{
    if (!image_path)
    {
        fprintf(stderr, "Error: No image path provided\n");
        return 1;
    }

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

    int num_rows = 0, num_cols = 0;
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
                grid_image.height, &y_boundaries, &x_boundaries, &num_rows,
                &num_cols))
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

        int crop_offset_x = 0, crop_offset_y = 0;
        if (!determine_text_region(
                &original_image, grid_bounds.x, grid_bounds.y,
                grid_bounds.width, grid_bounds.height, valid_letters,
                &text_region, &crop_offset_x, &crop_offset_y))
        {
            fprintf(stderr, "Failed to determine text region\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            return 1;
        }

        if (determine_grid_dimensions_from_letters(valid_letters, crop_offset_x,
                                                   crop_offset_y, &num_rows,
                                                   &num_cols) != 0)
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

        if (generate_cell_boundaries_from_letters(valid_letters, num_rows, num_cols,
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
        cpy_image(&grid_image, &text_region);
    }

    save_image("step_08_text_region.png", &text_region);

    if (create_button_callback)
    {
        create_button_callback("Text Region", "step_08_text_region.png");
    }

    Image debug_grid = {0};
    cpy_image(&text_region, &debug_grid);
    gray_to_rgba(&debug_grid);
    for (int i = 0; i <= num_rows; i++)
    {
        int y = y_boundaries[i];
        draw_rectangle(&debug_grid, 0, y, text_region.width, 1, true, 3,
                       0xFF0000FF);
    }

    for (int i = 0; i <= num_cols; i++)
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

    if (extract_cell_images(&text_region, y_boundaries, x_boundaries, num_rows,
                            num_cols) != 0)
    {
        fprintf(stderr, "Failed to extract cell images\n");
    }

    if (create_reconstructed_grid(num_rows, num_cols, create_button_callback) !=
        0)
    {
        fprintf(stderr, "Failed to create reconstructed grid\n");
    }

    printf("Extracted grid: %d rows x %d columns\n", num_rows, num_cols);

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

void clear_cells_directory(void)
{
    DIR *dir = opendir("cells");
    if (!dir)
    {
        printf("Could not open cells directory for clearing\n");
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
        snprintf(filepath, sizeof(filepath), "cells/%s", entry->d_name);
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
    printf("Cells directory cleared (except .gitkeep)\n");
}

int extract_cell_images(const Image *grid_region, const int *y_boundaries,
                        const int *x_boundaries, int num_rows, int num_cols)
{
    if (!grid_region || !y_boundaries || !x_boundaries)
    {
        return 1;
    }

    clear_cells_directory();

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

    free_image(&original_image);
    freeBoundingBoxArray(detected_words);
    free(detected_words);

    printf("Word detection processing completed\n");
    return 0;
}
