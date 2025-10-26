#include "./wordsearch_processor.h"
#include "../processing/image_preprocessing.h"
#include "../processing/contour_analysis.h"
#include "../grid/grid_analysis.h"
#include <stdio.h>
#include <stdlib.h>

int process_wordsearch_image(const char* image_path, CreateButtonCallback create_button_callback) {
    if (!image_path) {
        fprintf(stderr, "Error: No image path provided\n");
        return 1;
    }

    printf("Processing word search image: %s\n", image_path);

    Image image;
    if (!load_and_preprocess_image(image_path, &image, create_button_callback)) {
        fprintf(stderr, "Failed to load and preprocess image\n");
        return 1;
    }

    Image original_image;
    load_image(image_path, &original_image);

    // Extract the grid region
    Image grid_image;
    Rect grid_bounds;
    if (!extract_grid_region(&image, &original_image, &grid_image, &grid_bounds, create_button_callback)) {
        fprintf(stderr, "Failed to extract grid region\n");
        free_image(&image);
        free_image(&original_image);
        return 1;
    }

    // Process the grid for OCR
    if (!process_grid_for_ocr(&grid_image, create_button_callback)) {
        fprintf(stderr, "Failed to process grid for OCR\n");
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    // Find contours that could represent letters
    Contours* letters_contours = findContours(&grid_image, 0);
    if (!letters_contours) {
        fprintf(stderr, "Failed to find contours in grid image\n");
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    // Filter contours based on letter criteria
    Contours* valid_letters = filter_letter_contours(letters_contours);
    if (!valid_letters) {
        fprintf(stderr, "Failed to filter letter contours\n");
        freeContours(letters_contours);
        free_image(&image);
        free_image(&original_image);
        free_image(&grid_image);
        return 1;
    }

    int use_line_detection = has_proper_grid_lines(&grid_image);

    int num_rows = 0, num_cols = 0;
    int* y_boundaries = NULL;
    int* x_boundaries = NULL;

    if (use_line_detection) {
        printf("Using line-based detection for grid with inner contours\n");

        // Detect grid lines
        Image horizontal_lines = {0};
        Image vertical_lines = {0};
        detect_grid_lines(&grid_image, &horizontal_lines, &vertical_lines);

        save_image("step_09_horizontal_lines.png", &horizontal_lines);
        save_image("step_09_vertical_lines.png", &vertical_lines);

        // Create buttons for line detection steps if callback provided
        if (create_button_callback) {
            create_button_callback("Horizontal Lines", "step_09_horizontal_lines.png");
            create_button_callback("Vertical Lines", "step_09_vertical_lines.png");
        }

        // Extract cell boundaries from lines
        if (!extract_cell_boundaries_from_lines(&horizontal_lines, &vertical_lines,
                                              grid_image.width, grid_image.height,
                                              &y_boundaries, &x_boundaries, &num_rows, &num_cols)) {
            printf("Failed to extract cell boundaries from lines, falling back to letter-based detection\n");
            use_line_detection = 0;
            free_image(&horizontal_lines);
            free_image(&vertical_lines);
        } else {
            free_image(&horizontal_lines);
            free_image(&vertical_lines);
        }
    }

    Image text_region = {0};
    if (!use_line_detection) {
        printf("Using letter-based detection for grid without inner contours\n");

        // Determine the text region based on letter positions
        int crop_offset_x = 0, crop_offset_y = 0;
        if (!determine_text_region(&original_image, grid_bounds.x, grid_bounds.y,
                                  grid_bounds.width, grid_bounds.height,
                                  valid_letters, &text_region, &crop_offset_x, &crop_offset_y)) {
            fprintf(stderr, "Failed to determine text region\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            return 1;
        }

        // Determine grid dimensions from letters
        if (determine_grid_dimensions_from_letters(valid_letters, crop_offset_x, crop_offset_y,
                                                  &num_rows, &num_cols) != 0) {
            fprintf(stderr, "Failed to determine grid dimensions\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            free_image(&text_region);
            return 1;
        }

        // Generate safe cell boundaries
        if (generate_safe_cell_boundaries(&text_region, num_rows, num_cols,
                                        &y_boundaries, &x_boundaries) != 0) {
            fprintf(stderr, "Failed to generate cell boundaries\n");
            freeContours(letters_contours);
            freeContours(valid_letters);
            free_image(&image);
            free_image(&original_image);
            free_image(&grid_image);
            free_image(&text_region);
            return 1;
        }
    } else {
        // For line detection, use the grid_image as text_region
        cpy_image(&grid_image, &text_region);
    }

    save_image("step_08_text_region.png", &text_region);

    // Create button for text region step if callback provided
    if (create_button_callback) {
        create_button_callback("Text Region", "step_08_text_region.png");
    }

    Image debug_grid = {0};
    cpy_image(&text_region, &debug_grid);

    for (int i = 0; i <= num_rows; i++) {
        int y = y_boundaries[i];
        draw_rectangle(&debug_grid, 0, y, text_region.width, 1, true, 3, 0xFF00FF00);
    }

    for (int i = 0; i <= num_cols; i++) {
        int x = x_boundaries[i];
        draw_rectangle(&debug_grid, x, 0, 1, text_region.height, true, 3, 0xFF00FF00);
    }

    save_image("step_9_grid_region.png", &debug_grid);

    // Create button for grid region step if callback provided
    if (create_button_callback) {
        create_button_callback("Grid Region", "step_9_grid_region.png");
    }

    free_image(&debug_grid);

    // Extract individual cell images
    if (extract_cell_images(&text_region, y_boundaries, x_boundaries, num_rows, num_cols) != 0) {
        fprintf(stderr, "Failed to extract cell images\n");
    }

    // Create reconstructed grid from extracted cells
    if (create_reconstructed_grid(num_rows, num_cols, create_button_callback) != 0) {
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

int extract_cell_images(const Image* grid_region, const int* y_boundaries,
                       const int* x_boundaries, int num_rows, int num_cols) {
    if (!grid_region || !y_boundaries || !x_boundaries) {
        return 1;
    }

    printf("Extracting individual cells for OCR processing...\n");

    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            int y1 = y_boundaries[row];
            int y2 = y_boundaries[row + 1];
            int x1 = x_boundaries[col];
            int x2 = x_boundaries[col + 1];

            int cell_width_px = x2 - x1;
            int cell_height_px = y2 - y1;

            if (cell_width_px > 0 && cell_height_px > 0) {
                Image cell_roi = {0};
                extract_rectangle(grid_region, x1, y1, cell_width_px, cell_height_px, &cell_roi);

                char filename[256];
                sprintf(filename, "cells/cell_%d_%d.png", row, col);
                save_image(filename, &cell_roi);

                free_image(&cell_roi);
            }
        }
    }

    return 0;
}

int create_reconstructed_grid(int num_rows, int num_cols, CreateButtonCallback create_button_callback) {
    if (num_rows <= 0 || num_cols <= 0) {
        return 1;
    }

    // Load the first cell to get dimensions
    Image first_cell = {0};
    char first_filename[256];
    sprintf(first_filename, "cells/cell_0_0.png");

    load_image(first_filename, &first_cell);
    if (!first_cell.rgba_pixels && !first_cell.gray_pixels) {
        fprintf(stderr, "Failed to load first cell image for dimensions\n");
        return 1;
    }

    int cell_width = first_cell.width;
    int cell_height = first_cell.height;
    free_image(&first_cell);

    // Add spacing between cells
    int spacing = 5;
    int grid_width = num_cols * cell_width + (num_cols - 1) * spacing;
    int grid_height = num_rows * cell_height + (num_rows - 1) * spacing;

    // Create white background image
    Image reconstructed_grid = {0};
    reconstructed_grid.width = grid_width;
    reconstructed_grid.height = grid_height;
    reconstructed_grid.is_grayscale = false; // Color image

    size_t pixel_count = grid_width * grid_height;
    reconstructed_grid.rgba_pixels = (uint32_t*)malloc(pixel_count * sizeof(uint32_t));
    if (!reconstructed_grid.rgba_pixels) {
        fprintf(stderr, "Failed to allocate memory for reconstructed grid image\n");
        return 1;
    }

    // Fill with red color for bounds (0xFF0000FF = red)
    for (size_t i = 0; i < pixel_count; i++) {
        reconstructed_grid.rgba_pixels[i] = 0xFF0000FF; // Red background for bounds
    }

    // Load and place each cell
    for (int row = 0; row < num_rows; row++) {
        for (int col = 0; col < num_cols; col++) {
            char filename[256];
            sprintf(filename, "cells/cell_%d_%d.png", row, col);

            Image cell = {0};
            load_image(filename, &cell);
            if (!cell.rgba_pixels && !cell.gray_pixels) {
                fprintf(stderr, "Failed to load cell image: %s\n", filename);
                continue; // Skip missing cells
            }

            // Calculate position with spacing (spacing becomes red bounds)
            int dest_x = col * (cell_width + spacing);
            int dest_y = row * (cell_height + spacing);

            // Copy cell to reconstructed grid
            for (int y = 0; y < cell_height && (dest_y + y) < grid_height; y++) {
                for (int x = 0; x < cell_width && (dest_x + x) < grid_width; x++) {
                    int dest_idx = (dest_y + y) * grid_width + (dest_x + x);

                    if (cell.is_grayscale) {
                        // Convert grayscale to RGBA
                        uint8_t gray = cell.gray_pixels[y * cell.width + x];
                        reconstructed_grid.rgba_pixels[dest_idx] = (gray << 24) | (gray << 16) | (gray << 8) | gray;
                    } else {
                        // Copy RGBA pixel directly
                        reconstructed_grid.rgba_pixels[dest_idx] = cell.rgba_pixels[y * cell.width + x];
                    }
                }
            }

            free_image(&cell);
        }
    }

    // Save the reconstructed grid
    save_image("step_10_reconstructed_grid.png", &reconstructed_grid);

    // Create button for reconstructed grid step if callback provided
    if (create_button_callback) {
        create_button_callback("Reconstructed Grid", "step_10_reconstructed_grid.png");
    }

    free_image(&reconstructed_grid);
    return 0;
}

int determine_grid_dimensions_from_letters(Contours* valid_letters,
                                         int crop_offset_x, int crop_offset_y,
                                         int* num_rows, int* num_cols) {
    if (!valid_letters || !num_rows || !num_cols) {
        return 1;
    }

    // Translate contours to match the cropped text region coordinate system
    translateContours(valid_letters, -crop_offset_x, -crop_offset_y);

    #define ROW_TOLERANCE 10

    LetterRow* rows = NULL;
    int rows_capacity = 0;
    int actual_num_rows = 0;

    for (int i = 0; i < valid_letters->count; i++) {
        Rect rect;
        if (!boundingRect(&valid_letters->contours[i], &rect)) {
            continue;
        }

        int letter_y = rect.y;
        int found_row_idx = -1;

        for (int j = 0; j < actual_num_rows; j++) {
            if (abs(letter_y - rows[j].row_y) <= ROW_TOLERANCE) {
                found_row_idx = j;
                break;
            }
        }

        if (found_row_idx == -1) {
            if (actual_num_rows >= rows_capacity) {
                rows_capacity = rows_capacity == 0 ? 4 : rows_capacity * 2;
                rows = (LetterRow*)realloc(rows, sizeof(LetterRow) * rows_capacity);
                if (!rows) {
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

        LetterRow* row = &rows[found_row_idx];
        if (row->count >= row->capacity) {
            row->capacity = row->capacity == 0 ? 4 : row->capacity * 2;
            row->letters = (Rect*)realloc(row->letters, sizeof(Rect) * row->capacity);
            if (!row->letters) {
                fprintf(stderr, "Failed to allocate memory for letters in row\n");
                for (int k = 0; k < actual_num_rows; k++) {
                    free(rows[k].letters);
                }
                free(rows);
                return 1;
            }
        }

        row->letters[row->count++] = rect;
    }

    *num_cols = 0;
    for (int i = 0; i < actual_num_rows; i++) {
        if (rows[i].count > *num_cols) {
            *num_cols = rows[i].count;
        }
    }

    *num_rows = actual_num_rows;

    printf("Detected grid structure: %d rows x %d columns\n", *num_rows, *num_cols);

    for (int i = 0; i < actual_num_rows; i++) {
        free(rows[i].letters);
    }
    free(rows);

    return 0;
}

int generate_safe_cell_boundaries(const Image* grid_region, int num_rows, int num_cols,
                                 int** y_boundaries, int** x_boundaries) {
    if (!grid_region || !y_boundaries || !x_boundaries || num_rows <= 0 || num_cols <= 0) {
        return 1;
    }

    int cell_width = grid_region->width / num_cols;
    int cell_height = grid_region->height / num_rows;
    printf("Cell dimensions: %dx%d\n", cell_width, cell_height);

    Image binary_debug = {0};
    cpy_image(grid_region, &binary_debug);
    convert_to_grayscale(&binary_debug);

    double debug_otsu_threshold = threshold(&binary_debug, 255);
    if (debug_otsu_threshold < 0) {
        fprintf(stderr, "Failed to apply Otsu's thresholding for debug\n");
        free_image(&binary_debug);
        return 1;
    }

    double mean_val = 0.0;
    for (int i = 0; i < binary_debug.width * binary_debug.height; i++) {
        mean_val += binary_debug.gray_pixels[i];
    }
    mean_val /= (binary_debug.width * binary_debug.height);

    if (mean_val > 127) {
        for (int i = 0; i < binary_debug.width * binary_debug.height; i++) {
            binary_debug.gray_pixels[i] = 255 - binary_debug.gray_pixels[i];
        }
    }

    *y_boundaries = (int*)malloc(sizeof(int) * (num_rows + 1));
    *x_boundaries = (int*)malloc(sizeof(int) * (num_cols + 1));
    if (!*y_boundaries || !*x_boundaries) {
        fprintf(stderr, "Failed to allocate memory for boundary arrays\n");
        free_image(&binary_debug);
        if (*y_boundaries) free(*y_boundaries);
        if (*x_boundaries) free(*x_boundaries);
        return 1;
    }

    for (int row = 0; row <= num_rows; row++) {
        int y = row * cell_height;
        int safe_y = find_safe_line_position(y, 1, 10, &binary_debug,
                                           grid_region->width, grid_region->height);
        (*y_boundaries)[row] = safe_y;
    }

    for (int col = 0; col <= num_cols; col++) {
        int x = col * cell_width;
        int safe_x = find_safe_line_position(x, 0, 10, &binary_debug,
                                           grid_region->width, grid_region->height);
        (*x_boundaries)[col] = safe_x;
    }

    free_image(&binary_debug);
    return 0;
}
