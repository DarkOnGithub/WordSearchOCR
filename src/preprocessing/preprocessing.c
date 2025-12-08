#include "processing/preprocessing.h"
#include "analysis/contour_analysis.h"
#include "analysis/grid_analysis.h"
#include <stdio.h>

int load_and_preprocess_image(const char *image_path, Image *image,
                              CreateButtonCallback create_button_callback)
{
    if (!image_path || !image)
    {
        return 0;
    }

    load_image(image_path, image);
    if (!image)
    {
        printf("Failed to load image: %s\n", image_path);
        return 0;
    }

    convert_to_grayscale(image);
    if (!rotate_image_automatic(image))
    {
        printf("Warning: Automatic rotation failed or was skipped\n");
    }
    save_image("step_01_grayscale.png", image);
    if (create_button_callback)
    {
        create_button_callback("Grayscale", "step_01_grayscale.png");
    }

    adaptive_denoise(image);
    save_image("step_02_adaptive_denoise.png", image);
    if (create_button_callback)
    {
        create_button_callback("Adaptive Denoise",
                               "step_02_adaptive_denoise.png");
    }

    adaptiveThreshold(image, 255, 1, 1, 11, 2.0);
    save_image("step_03_threshold.png", image);
    if (create_button_callback)
    {
        create_button_callback("Threshold", "step_03_threshold.png");
    }

    adaptive_morphological_clean(image);
    save_image("step_03_5_morph_cleaned.png", image);
    return 1;
}

int extract_grid_region(const Image *processed_image,
                        const Image *original_image, Image *grid_image,
                        Rect *grid_bounds,
                        CreateButtonCallback create_button_callback)
{
    if (!processed_image || !original_image || !grid_image)
    {
        return 0;
    }

    Image horizontal_lines = {0};
    Image vertical_lines = {0};
    detect_grid_lines(processed_image, &horizontal_lines, &vertical_lines);

    int grid_detected = 0;
    if (horizontal_lines.width > 0 && horizontal_lines.height > 0 &&
        vertical_lines.width > 0 && vertical_lines.height > 0)
    {
        Contours *horiz_contours = findContours(&horizontal_lines, 0);
        Contours *vert_contours = findContours(&vertical_lines, 0);

        if (horiz_contours && vert_contours &&
            horiz_contours->count >= 2 && vert_contours->count >= 2)
        {
            double min_horiz_length = processed_image->width * 0.4;
            double min_vert_length = processed_image->height * 0.4;

            Contours *filtered_horiz =
                filterContoursByLength(horiz_contours, min_horiz_length);
            Contours *filtered_vert =
                filterContoursByLength(vert_contours, min_vert_length);

            freeContours(horiz_contours);
            freeContours(vert_contours);

            if (filtered_horiz && filtered_vert &&
                filtered_horiz->count >= 2 && filtered_vert->count >= 2)
            {
                printf("Detected grid lines: %d horizontal, %d vertical\n",
                       filtered_horiz->count, filtered_vert->count);

                // Get bounding rectangles for all filtered contours to find overall grid bounds
                Rect *horiz_rects = (Rect *)malloc(sizeof(Rect) * filtered_horiz->count);
                Rect *vert_rects = (Rect *)malloc(sizeof(Rect) * filtered_vert->count);

                if (horiz_rects && vert_rects)
                {
                    int valid_horiz = 0, valid_vert = 0;

                    // Get bounding rects for horizontal lines
                    for (int i = 0; i < filtered_horiz->count; i++)
                    {
                        if (boundingRect(&filtered_horiz->contours[i], &horiz_rects[i]))
                        {
                            valid_horiz++;
                        }
                    }

                    // Get bounding rects for vertical lines
                    for (int i = 0; i < filtered_vert->count; i++)
                    {
                        if (boundingRect(&filtered_vert->contours[i], &vert_rects[i]))
                        {
                            valid_vert++;
                        }
                    }

                    if (valid_horiz >= 2 && valid_vert >= 2)
                    {
                        // Find the overall grid bounds by taking min/max of all line positions
                        int min_x = processed_image->width;
                        int max_x = 0;
                        int min_y = processed_image->height;
                        int max_y = 0;

                        // Use vertical lines for X bounds (leftmost and rightmost)
                        for (int i = 0; i < valid_vert; i++)
                        {
                            if (vert_rects[i].x < min_x) min_x = vert_rects[i].x;
                            if (vert_rects[i].x + vert_rects[i].width > max_x)
                                max_x = vert_rects[i].x + vert_rects[i].width;
                        }

                        // Use horizontal lines for Y bounds (topmost and bottommost)
                        for (int i = 0; i < valid_horiz; i++)
                        {
                            if (horiz_rects[i].y < min_y) min_y = horiz_rects[i].y;
                            if (horiz_rects[i].y + horiz_rects[i].height > max_y)
                                max_y = horiz_rects[i].y + horiz_rects[i].height;
                        }

                        // Add some padding around the detected grid
                        int padding = 5; // pixels
                        min_x = (min_x - padding > 0) ? min_x - padding : 0;
                        min_y = (min_y - padding > 0) ? min_y - padding : 0;
                        max_x = (max_x + padding < processed_image->width) ? max_x + padding : processed_image->width;
                        max_y = (max_y + padding < processed_image->height) ? max_y + padding : processed_image->height;

                        grid_bounds->x = min_x;
                        grid_bounds->y = min_y;
                        grid_bounds->width = max_x - min_x;
                        grid_bounds->height = max_y - min_y;

                        printf("Grid lines bounding rect: x=%d, y=%d, w=%d, h=%d\n",
                               min_x, min_y, grid_bounds->width, grid_bounds->height);

                        grid_detected = 1;
                    }
                }

                if (horiz_rects) free(horiz_rects);
                if (vert_rects) free(vert_rects);
            }

            if (filtered_horiz) freeContours(filtered_horiz);
            if (filtered_vert) freeContours(filtered_vert);
        }
        else
        {
            if (horiz_contours) freeContours(horiz_contours);
            if (vert_contours) freeContours(vert_contours);
        }
    }

    free_image(&horizontal_lines);
    free_image(&vertical_lines);

    // If grid lines weren't detected or failed, fall back to contour method
    if (!grid_detected)
    {
        printf("Grid lines not detected, falling back to contour method\n");

        // Find contours in the processed binary image
        Contours *grid_contours = findContours(processed_image, 0);
        if (!grid_contours || grid_contours->count == 0)
        {
            printf("No contours found in processed image\n");
            if (grid_contours)
                freeContours(grid_contours);
            return 0;
        }

        // Find the largest contour (assumed to be the grid boundary)
        Contour *best_contour = find_largest_contour(grid_contours);
        if (!best_contour)
        {
            printf("Failed to find largest contour\n");
            freeContours(grid_contours);
            return 0;
        }

        // Get bounding rectangle of the grid
        if (!boundingRect(best_contour, grid_bounds))
        {
            printf("Failed to get bounding rectangle for grid contour\n");
            freeContours(grid_contours);
            return 0;
        }

        freeContours(grid_contours);

        int x = grid_bounds->x, y = grid_bounds->y;
        int w = grid_bounds->width, h = grid_bounds->height;
        int area = w * h;

        printf("Grid contour bounding rect: x=%d, y=%d, w=%d, h=%d (area=%d)\n", x,
               y, w, h, area);
    }

    int x = grid_bounds->x, y = grid_bounds->y;
    int w = grid_bounds->width, h = grid_bounds->height;

    extract_rectangle(original_image, x, y, w, h, grid_image);
    convert_to_grayscale(grid_image);
    save_image("step_05_grid_extraction.png", grid_image);
    if (create_button_callback)
    {
        create_button_callback("Grid Extraction",
                               "step_05_grid_extraction.png");
    }

    return 1;
}

int process_grid_for_ocr(Image *grid_image,
                         CreateButtonCallback create_button_callback)
{
    if (!grid_image)
    {
        return 0;
    }

    double otsu_threshold = threshold(grid_image, 255);
    if (otsu_threshold < 0)
    {
        printf("Failed to apply Otsu's thresholding\n");
        return 0;
    }

    if (correctBinaryImageOrientation(grid_image) < 0)
    {
        printf("Failed to correct binary image orientation\n");
        return 0;
    }

    save_image("step_06_binary_grid.png", grid_image);
    // if (create_button_callback) {
    //     create_button_callback("Binary Grid", "step_06_binary_grid.png");
    // }

    StructuringElement *cleanup_kernel = getStructuringElement(0, 2, 2);
    if (cleanup_kernel)
    {
        morphologyEx(grid_image, MORPH_CLOSE, cleanup_kernel, 1);
        freeStructuringElement(cleanup_kernel);
        save_image("step_07_cleaned_grid.png", grid_image);
        if (create_button_callback)
        {
            create_button_callback("Cleaned Grid", "step_07_cleaned_grid.png");
        }
    }

    return 1;
}

int determine_text_region(const Image *original_image, int grid_x, int grid_y,
                          int grid_width, int grid_height,
                          const Contours *valid_letters, Image *text_region,
                          int *crop_offset_x, int *crop_offset_y)
{
    if (!original_image || !text_region)
    {
        return 0;
    }

    if (!valid_letters || valid_letters->count == 0)
    {
        // No letters found, use full grid region
        extract_rectangle(original_image, grid_x, grid_y, grid_width,
                          grid_height, text_region);
        convert_to_grayscale(text_region);
        if (crop_offset_x)
            *crop_offset_x = 0;
        if (crop_offset_y)
            *crop_offset_y = 0;
        printf("No letters found, using full grid region\n");
        return 1;
    }

    Rect *letter_rects = (Rect *)malloc(sizeof(Rect) * valid_letters->count);
    if (!letter_rects)
    {
        printf("Failed to allocate memory for letter rectangles\n");
        return 0;
    }

    for (int i = 0; i < valid_letters->count; i++)
    {
        if (!boundingRect(&valid_letters->contours[i], &letter_rects[i]))
        {
            printf("Failed to get bounding rectangle for letter %d\n", i);
            free(letter_rects);
            return 0;
        }
    }

    Rect text_bounds;
    if (getBoundingRectOfRects(letter_rects, valid_letters->count, 5,
                               grid_width, grid_height, &text_bounds))
    {
        int translated_x = grid_x + text_bounds.x;
        int translated_y = grid_y + text_bounds.y;

        extract_rectangle(original_image, translated_x, translated_y,
                          text_bounds.width, text_bounds.height, text_region);
        convert_to_grayscale(text_region);

        if (crop_offset_x)
            *crop_offset_x = text_bounds.x;
        if (crop_offset_y)
            *crop_offset_y = text_bounds.y;

        printf("Letter-based detection: cropped text region: %dx%d at (%d,%d) "
               "[translated from grid coords (%d,%d)]\n",
               text_bounds.width, text_bounds.height, translated_x,
               translated_y, text_bounds.x, text_bounds.y);
    }
    else
    {
        printf("Failed to calculate text bounds, using full grid region\n");
        extract_rectangle(original_image, grid_x, grid_y, grid_width,
                          grid_height, text_region);
        convert_to_grayscale(text_region);
        if (crop_offset_x)
            *crop_offset_x = 0;
        if (crop_offset_y)
            *crop_offset_y = 0;
    }

    free(letter_rects);
    return 1;
}
