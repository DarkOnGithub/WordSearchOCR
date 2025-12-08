#include "wordsearch/word_detection.h"
#include "image/image.h"
#include "image/operations.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void initBoundingBoxArray(BoundingBoxArray *array)
{
    array->boxes = NULL;
    array->count = 0;
    array->capacity = 0;
}

bool addBoundingBox(BoundingBoxArray *array, BoundingBox box)
{
    if (array->count >= array->capacity)
    {
        int new_capacity = array->capacity == 0 ? 16 : array->capacity * 2;
        BoundingBox *new_boxes =
            realloc(array->boxes, new_capacity * sizeof(BoundingBox));
        if (!new_boxes)
        {
            return false;
        }
        array->boxes = new_boxes;
        array->capacity = new_capacity;
    }

    array->boxes[array->count++] = box;
    return true;
}

void freeBoundingBoxArray(BoundingBoxArray *array)
{
    if (array->boxes)
    {
        free(array->boxes);
        array->boxes = NULL;
    }
    array->count = 0;
    array->capacity = 0;
}

void initWordGroups(WordGroups *groups)
{
    groups->groups = NULL;
    groups->count = 0;
    groups->capacity = 0;
}

bool addWordGroup(WordGroups *groups, BoundingBoxArray *group)
{
    if (groups->count >= groups->capacity)
    {
        int new_capacity = groups->capacity == 0 ? 4 : groups->capacity * 2;
        BoundingBoxArray **new_groups =
            realloc(groups->groups, new_capacity * sizeof(BoundingBoxArray *));
        if (!new_groups)
        {
            return false;
        }
        groups->groups = new_groups;
        groups->capacity = new_capacity;
    }

    groups->groups[groups->count++] = group;
    return true;
}

void freeWordGroups(WordGroups *groups)
{
    if (groups->groups)
    {
        for (int i = 0; i < groups->count; i++)
        {
            freeBoundingBoxArray(groups->groups[i]);
            free(groups->groups[i]);
        }
        free(groups->groups);
        groups->groups = NULL;
    }
    groups->count = 0;
    groups->capacity = 0;
}

/*
    Merge nearby bounding boxes based on gap and vertical overlap criteria
*/
BoundingBoxArray *merge_nearby_boxes(const BoundingBoxArray *boxes, int max_gap,
                                     float max_vertical_overlap)
{
    if (!boxes || boxes->count == 0)
    {
        BoundingBoxArray *result = malloc(sizeof(BoundingBoxArray));
        if (result)
        {
            initBoundingBoxArray(result);
        }
        return result;
    }

    BoundingBoxArray *sorted_boxes = malloc(sizeof(BoundingBoxArray));
    if (!sorted_boxes)
    {
        return NULL;
    }
    initBoundingBoxArray(sorted_boxes);

    for (int i = 0; i < boxes->count; i++)
    {
        addBoundingBox(sorted_boxes, boxes->boxes[i]);
    }

    for (int i = 0; i < sorted_boxes->count - 1; i++)
    {
        for (int j = 0; j < sorted_boxes->count - i - 1; j++)
        {
            if (sorted_boxes->boxes[j].x > sorted_boxes->boxes[j + 1].x)
            {
                BoundingBox temp = sorted_boxes->boxes[j];
                sorted_boxes->boxes[j] = sorted_boxes->boxes[j + 1];
                sorted_boxes->boxes[j + 1] = temp;
            }
        }
    }

    BoundingBoxArray *merged_boxes = malloc(sizeof(BoundingBoxArray));
    if (!merged_boxes)
    {
        freeBoundingBoxArray(sorted_boxes);
        free(sorted_boxes);
        return NULL;
    }
    initBoundingBoxArray(merged_boxes);

    BoundingBox current_box = sorted_boxes->boxes[0];

    for (int i = 1; i < sorted_boxes->count; i++)
    {
        BoundingBox box = sorted_boxes->boxes[i];
        int x = box.x;
        int y = box.y;
        int w = box.width;
        int h = box.height;

        int cx1 = current_box.x;
        int cy1 = current_box.y;
        int cw1 = current_box.width;
        int ch1 = current_box.height;

        int overlap_start = (cy1 > y) ? cy1 : y;
        int overlap_end = (cy1 + ch1 < y + h) ? (cy1 + ch1) : (y + h);
        int y_overlap =
            (overlap_end > overlap_start) ? (overlap_end - overlap_start) : 0;
        int max_h = (ch1 > h) ? ch1 : h;
        float overlap_ratio = max_h > 0 ? (float)y_overlap / max_h : 0.0f;

        int gap = x - (cx1 + cw1);

        if (gap <= max_gap && overlap_ratio > max_vertical_overlap)
        {
            int new_x = (cx1 < x) ? cx1 : x;
            int new_y = (cy1 < y) ? cy1 : y;
            int new_w = ((cx1 + cw1 > x + w) ? (cx1 + cw1) : (x + w)) - new_x;
            int new_h = ((cy1 + ch1 > y + h) ? (cy1 + ch1) : (y + h)) - new_y;
            current_box.x = new_x;
            current_box.y = new_y;
            current_box.width = new_w;
            current_box.height = new_h;
        }
        else
        {
            addBoundingBox(merged_boxes, current_box);
            current_box = box;
        }
    }

    addBoundingBox(merged_boxes, current_box);

    freeBoundingBoxArray(sorted_boxes);
    free(sorted_boxes);

    return merged_boxes;
}


WordGroups *find_word_groups(const BoundingBoxArray *boxes, int max_distance,
                             int alignment_threshold)
{
    printf("find_word_groups: processing %d boxes\n", boxes ? boxes->count : 0);
    if (!boxes || boxes->count == 0)
    {
        WordGroups *result = malloc(sizeof(WordGroups));
        if (result)
        {
            initWordGroups(result);
        }
        return result;
    }

    BoundingBoxArray *sorted_boxes = malloc(sizeof(BoundingBoxArray));
    if (!sorted_boxes)
    {
        return NULL;
    }
    initBoundingBoxArray(sorted_boxes);

    for (int i = 0; i < boxes->count; i++)
    {
        addBoundingBox(sorted_boxes, boxes->boxes[i]);
    }

    for (int i = 0; i < sorted_boxes->count - 1; i++)
    {
        for (int j = 0; j < sorted_boxes->count - i - 1; j++)
        {
            if (sorted_boxes->boxes[j].y > sorted_boxes->boxes[j + 1].y)
            {
                BoundingBox temp = sorted_boxes->boxes[j];
                sorted_boxes->boxes[j] = sorted_boxes->boxes[j + 1];
                sorted_boxes->boxes[j + 1] = temp;
            }
        }
    }

    WordGroups *rows = malloc(sizeof(WordGroups));
    if (!rows)
    {
        freeBoundingBoxArray(sorted_boxes);
        free(sorted_boxes);
        return NULL;
    }
    initWordGroups(rows);

    if (sorted_boxes->count > 0)
    {
        BoundingBoxArray *current_row = malloc(sizeof(BoundingBoxArray));
        if (!current_row)
        {
            freeBoundingBoxArray(sorted_boxes);
            free(sorted_boxes);
            freeWordGroups(rows);
            free(rows);
            return NULL;
        }
        initBoundingBoxArray(current_row);
        addBoundingBox(current_row, sorted_boxes->boxes[0]);

        float current_y =
            sorted_boxes->boxes[0].y + sorted_boxes->boxes[0].height / 2.0f;

        for (int i = 1; i < sorted_boxes->count; i++)
        {
            BoundingBox box = sorted_boxes->boxes[i];
            float box_y = box.y + box.height / 2.0f;

            if (fabsf(box_y - current_y) <= alignment_threshold)
            {
                addBoundingBox(current_row, box);
            }
            else
            {
                for (int j = 0; j < current_row->count - 1; j++)
                {
                    for (int k = 0; k < current_row->count - j - 1; k++)
                    {
                        if (current_row->boxes[k].x >
                            current_row->boxes[k + 1].x)
                        {
                            BoundingBox temp = current_row->boxes[k];
                            current_row->boxes[k] = current_row->boxes[k + 1];
                            current_row->boxes[k + 1] = temp;
                        }
                    }
                }
                addWordGroup(rows, current_row);

                current_row = malloc(sizeof(BoundingBoxArray));
                if (!current_row)
                {
                    freeBoundingBoxArray(sorted_boxes);
                    free(sorted_boxes);
                    freeWordGroups(rows);
                    free(rows);
                    return NULL;
                }
                initBoundingBoxArray(current_row);
                addBoundingBox(current_row, box);
                current_y = box_y;
            }
        }

        if (current_row->count > 0)
        {
            for (int j = 0; j < current_row->count - 1; j++)
            {
                for (int k = 0; k < current_row->count - j - 1; k++)
                {
                    if (current_row->boxes[k].x > current_row->boxes[k + 1].x)
                    {
                        BoundingBox temp = current_row->boxes[k];
                        current_row->boxes[k] = current_row->boxes[k + 1];
                        current_row->boxes[k + 1] = temp;
                    }
                }
            }
            addWordGroup(rows, current_row);
        }
        else
        {
            freeBoundingBoxArray(current_row);
            free(current_row);
        }
    }

    freeBoundingBoxArray(sorted_boxes);
    free(sorted_boxes);

    printf("find_word_groups: found %d rows\n", rows->count);
    for (int i = 0; i < rows->count; i++)
    {
        printf("  Row %d: %d boxes\n", i, rows->groups[i]->count);
    }

    if (rows->count < 2)
    {
        return rows;
    }

    float *vertical_gaps = malloc((rows->count - 1) * sizeof(float));
    if (!vertical_gaps)
    {
        freeWordGroups(rows);
        free(rows);
        return NULL;
    }

    for (int i = 0; i < rows->count - 1; i++)
    {
        BoundingBoxArray *row1 = rows->groups[i];
        BoundingBoxArray *row2 = rows->groups[i + 1];

        float row1_y = 0.0f;
        for (int j = 0; j < row1->count; j++)
        {
            row1_y += row1->boxes[j].y + row1->boxes[j].height / 2.0f;
        }
        row1_y /= row1->count;

        float row2_y = 0.0f;
        for (int j = 0; j < row2->count; j++)
        {
            row2_y += row2->boxes[j].y + row2->boxes[j].height / 2.0f;
        }
        row2_y /= row2->count;

        vertical_gaps[i] = row2_y - row1_y;
    }

    int *gap_counts = calloc(
        1000,
        sizeof(int)); // The gap shouldn't be more than 1000 pixels (I hope :))
    if (!gap_counts)
    {
        free(vertical_gaps);
        freeWordGroups(rows);
        free(rows);
        return NULL;
    }

    for (int i = 0; i < rows->count - 1; i++)
    {
        int rounded_gap = (int)roundf(vertical_gaps[i] / 10.0f) * 10;
        if (rounded_gap >= 0 && rounded_gap < 1000)
        {
            gap_counts[rounded_gap]++;
        }
    }

    int most_common_gap = 0;
    int max_count = 0;
    for (int i = 0; i < 1000; i++)
    {
        if (gap_counts[i] > max_count)
        {
            max_count = gap_counts[i];
            most_common_gap = i;
        }
    }

    free(gap_counts);
    free(vertical_gaps);

    WordGroups *groups = malloc(sizeof(WordGroups));
    if (!groups)
    {
        freeWordGroups(rows);
        free(rows);
        return NULL;
    }
    initWordGroups(groups);

    if (rows->count > 0)
    {
        BoundingBoxArray *current_group = malloc(sizeof(BoundingBoxArray));
        if (!current_group)
        {
            freeWordGroups(rows);
            free(rows);
            freeWordGroups(groups);
            free(groups);
            return NULL;
        }
        initBoundingBoxArray(current_group);

        for (int j = 0; j < rows->groups[0]->count; j++)
        {
            addBoundingBox(current_group, rows->groups[0]->boxes[j]);
        }

        for (int i = 1; i < rows->count; i++)
        {
            BoundingBoxArray *prev_row = rows->groups[i - 1];
            BoundingBoxArray *current_row = rows->groups[i];

            float prev_row_y = 0.0f;
            for (int j = 0; j < prev_row->count; j++)
            {
                prev_row_y +=
                    prev_row->boxes[j].y + prev_row->boxes[j].height / 2.0f;
            }
            prev_row_y /= prev_row->count;

            float current_row_y = 0.0f;
            for (int j = 0; j < current_row->count; j++)
            {
                current_row_y += current_row->boxes[j].y +
                                 current_row->boxes[j].height / 2.0f;
            }
            current_row_y /= current_row->count;

            float gap = current_row_y - prev_row_y;
            int rounded_gap = (int)roundf(gap / 10.0f) * 10;

            if (abs(rounded_gap - most_common_gap) <= 20)
            {
                for (int j = 0; j < current_row->count; j++)
                {
                    addBoundingBox(current_group, current_row->boxes[j]);
                }
            }
            else
            {
                addWordGroup(groups, current_group);

                current_group = malloc(sizeof(BoundingBoxArray));
                if (!current_group)
                {
                    freeWordGroups(rows);
                    free(rows);
                    freeWordGroups(groups);
                    free(groups);
                    return NULL;
                }
                initBoundingBoxArray(current_group);

                for (int j = 0; j < current_row->count; j++)
                {
                    addBoundingBox(current_group, current_row->boxes[j]);
                }
            }
        }

        if (current_group->count > 0)
        {
            addWordGroup(groups, current_group);
        }
        else
        {
            freeBoundingBoxArray(current_group);
            free(current_group);
        }
    }

    freeWordGroups(rows);
    free(rows);

    return groups;
}

/*
    Select the main word group (largest group with at least 2 words)
*/
BoundingBoxArray *select_main_word_group(const WordGroups *groups)
{
    if (!groups || groups->count == 0)
    {
        return NULL;
    }

    int largest_index = 0;
    int largest_size = 0;

    for (int i = 0; i < groups->count; i++)
    {
        if (groups->groups[i]->count > largest_size)
        {
            largest_size = groups->groups[i]->count;
            largest_index = i;
        }
    }

    if (largest_size >= 2)
    {
        BoundingBoxArray *result = malloc(sizeof(BoundingBoxArray));
        if (!result)
        {
            return NULL;
        }
        initBoundingBoxArray(result);

        BoundingBoxArray *largest_group = groups->groups[largest_index];
        for (int i = 0; i < largest_group->count; i++)
        {
            addBoundingBox(result, largest_group->boxes[i]);
        }

        return result;
    }
    else
    {
        return NULL;
    }
}

void draw_bounding_boxes(Image *image, const BoundingBoxArray *boxes,
                         uint32_t color, int thickness)
{
    if (!image || !boxes)
    {
        return;
    }

    for (int i = 0; i < boxes->count; i++)
    {
        BoundingBox box = boxes->boxes[i];
        draw_rectangle(image, box.x, box.y, box.width, box.height, false,
                       thickness, color);
    }
}

/*
    Resize a grayscale image to fit within target dimensions while maintaining aspect ratio.
    The image will be centered in the target frame with padding if needed.
*/
void resize_letter_maintaining_aspect_ratio(Image *src, Image *dst, int target_width, int target_height)
{
    if (!src || !dst || !src->is_grayscale || !src->gray_pixels)
    {
        fprintf(stderr, "Error: Invalid source image for letter resizing\n");
        return;
    }

    dst->width = target_width;
    dst->height = target_height;
    dst->is_grayscale = true;
    dst->rgba_pixels = NULL;
    dst->gray_pixels = (uint8_t*)malloc(target_width * target_height * sizeof(uint8_t));

    if (!dst->gray_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate memory for resized letter image\n");
        return;
    }

    memset(dst->gray_pixels, 255, target_width * target_height * sizeof(uint8_t));

    float scale_x = (float)target_width / src->width;
    float scale_y = (float)target_height / src->height;
    float scale = (scale_x < scale_y) ? scale_x : scale_y;

    int new_width = (int)(src->width * scale);
    int new_height = (int)(src->height * scale);

    int offset_x = (target_width - new_width) / 2;
    int offset_y = (target_height - new_height) / 2;

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            float src_x = x / scale;
            float src_y = y / scale;

            int x1 = (int)floorf(src_x);
            int y1 = (int)floorf(src_y);
            int x2 = (int)ceilf(src_x);
            int y2 = (int)ceilf(src_y);

            x1 = x1 < 0 ? 0 : (x1 >= src->width ? src->width - 1 : x1);
            x2 = x2 < 0 ? 0 : (x2 >= src->width ? src->width - 1 : x2);
            y1 = y1 < 0 ? 0 : (y1 >= src->height ? src->height - 1 : y1);
            y2 = y2 < 0 ? 0 : (y2 >= src->height ? src->height - 1 : y2);

            uint8_t p11 = src->gray_pixels[y1 * src->width + x1];
            uint8_t p12 = src->gray_pixels[y1 * src->width + x2];
            uint8_t p21 = src->gray_pixels[y2 * src->width + x1];
            uint8_t p22 = src->gray_pixels[y2 * src->width + x2];

            float dx = src_x - x1;
            float dy = src_y - y1;

            float interpolated = p11 * (1 - dx) * (1 - dy) +
                               p12 * dx * (1 - dy) +
                               p21 * (1 - dx) * dy +
                               p22 * dx * dy;

            dst->gray_pixels[(offset_y + y) * target_width + (offset_x + x)] = (uint8_t)roundf(interpolated);
        }
    }
}

/*
    Extract individual letters from a word image and return them as a tensor of 28x28 images.
    Each letter maintains its aspect ratio without stretching.
*/
Tensor *extract_word_letters(const Image *word_image)
{
    if (!word_image || !word_image->gray_pixels)
    {
        fprintf(stderr, "Error: Invalid word image for letter extraction\n");
        return NULL;
    }

    Image processed_image;
    cpy_image(word_image, &processed_image);

    if (!processed_image.is_grayscale)
    {
        convert_to_grayscale(&processed_image);
    }

    threshold(&processed_image, 255);
    invert(&processed_image);

    Contours *contours = findContours(&processed_image, 0); // RETR_EXTERNAL
    if (!contours)
    {
        free_image(&processed_image);
        return NULL;
    }

    int letter_count = 0;
    Rect *letter_rects = NULL;

    for (int i = 0; i < contours->count; i++)
    {
        Rect bounding_rect;
        if (!boundingRect(&contours->contours[i], &bounding_rect))
        {
            continue;
        }

        if (bounding_rect.width > 2 && bounding_rect.height > 10)
        {
            letter_rects = realloc(letter_rects, (letter_count + 1) * sizeof(Rect));
            if (!letter_rects)
            {
                freeContours(contours);
                free_image(&processed_image);
                return NULL;
            }
            letter_rects[letter_count++] = bounding_rect;
        }
    }

    freeContours(contours);
    printf("Found %d letters\n", letter_count);
    if (letter_count == 0)
    {
        free_image(&processed_image);
        return NULL;
    }

    for (int i = 0; i < letter_count - 1; i++)
    {
        for (int j = 0; j < letter_count - i - 1; j++)
        {
            if (letter_rects[j].x > letter_rects[j + 1].x)
            {
                Rect temp = letter_rects[j];
                letter_rects[j] = letter_rects[j + 1];
                letter_rects[j + 1] = temp;
            }
        }
    }

    int tensor_shape[3] = {letter_count, 28, 28};
    Tensor *letters_tensor = tensor_create(tensor_shape, 3);
    if (!letters_tensor)
    {
        free(letter_rects);
        free_image(&processed_image);
        return NULL;
    }

    for (int i = 0; i < letter_count; i++)
    {
        Rect letter_rect = letter_rects[i];

        Image letter_image;
        extract_rectangle(word_image, letter_rect.x, letter_rect.y,
                         letter_rect.width, letter_rect.height, &letter_image);
        Image resized_letter;
        resize_letter_maintaining_aspect_ratio(&letter_image, &resized_letter, 28, 28);

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                float pixel_value = resized_letter.gray_pixels[y * 28 + x] / 255.0f;
                letters_tensor->data[i * 28 * 28 + y * 28 + x] = pixel_value;
            }
        }

        free_image(&letter_image);
        free_image(&resized_letter);
    }

    free(letter_rects);
    free_image(&processed_image);

    return letters_tensor;
}

BoundingBoxArray *detect_words(const char *image_path, const char *debug_prefix)
{
    Image image;
    load_image(image_path, &image);
    if (!rotate_image_automatic(&image))
    {
        printf("Warning: Automatic rotation for word detection failed or was skipped\n");
    }

    if (!image.rgba_pixels && !image.gray_pixels)
    {
        fprintf(stderr, "Error: Could not load image from %s\n", image_path);
        return NULL;
    }

    if (!image.is_grayscale)
    {
        convert_to_grayscale(&image);
    }

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_01_original.png",
                 debug_prefix);
        save_image(debug_path, &image);
    }

    gaussian_blur(&image, 5, 0.0);

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_02_blurred.png",
                 debug_prefix);
        save_image(debug_path, &image);
    }

    threshold(&image, 255);
    correctBinaryImageOrientation(&image);

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_03_otsu_threshold.png",
                 debug_prefix);
        save_image(debug_path, &image);
    }

    Image original_blurred2;
    load_image(image_path, &original_blurred2);
    if (!rotate_image_automatic(&original_blurred2))
    {
        printf("Warning: Automatic rotation for secondary blur failed or was skipped\n");
    }
    convert_to_grayscale(&original_blurred2);
    gaussian_blur(&original_blurred2, 5, 0.0);

    int total_pixels = original_blurred2.width * original_blurred2.height;
    double mean_val = 0.0;
    for (int i = 0; i < total_pixels; i++)
    {
        mean_val += original_blurred2.gray_pixels[i];
    }
    mean_val /= total_pixels;

    for (int i = 0; i < total_pixels; i++)
    {
        original_blurred2.gray_pixels[i] =
            (original_blurred2.gray_pixels[i] > mean_val * 0.8) ? 0 : 255;
    }

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_05_mean_threshold.png",
                 debug_prefix);
        save_image(debug_path, &image);
    }

    Image combined_threshold;
    combined_threshold.width = image.width;
    combined_threshold.height = image.height;
    combined_threshold.is_grayscale = true;
    combined_threshold.rgba_pixels = NULL;
    combined_threshold.gray_pixels = malloc(total_pixels * sizeof(uint8_t));

    if (!combined_threshold.gray_pixels)
    {
        fprintf(stderr,
                "Error: Failed to allocate memory for combined threshold\n");
        free_image(&image);
        free_image(&original_blurred2);
        return NULL;
    }

    for (int i = 0; i < total_pixels; i++)
    {
        uint8_t val1 = image.gray_pixels[i];
        uint8_t val3 = original_blurred2.gray_pixels[i]; // Mean result
        combined_threshold.gray_pixels[i] = val1 | val3;
    }

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_06_combined_threshold.png",
                 debug_prefix);
        save_image(debug_path, &combined_threshold);
    }

    free_image(&original_blurred2);

    StructuringElement *kernel_small =
        getStructuringElement(0, 2, 2); // MORPH_RECT, 2x2
    StructuringElement *kernel_large =
        getStructuringElement(0, 3, 3); // MORPH_RECT, 3x3

    if (!kernel_small || !kernel_large)
    {
        fprintf(stderr, "Error: Failed to create structuring elements\n");
        free_image(&image);
        free_image(&combined_threshold);
        return NULL;
    }

    morphologyEx(&combined_threshold, MORPH_CLOSE, kernel_large, 2);

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_07_morphology_closed.png",
                 debug_prefix);
        save_image(debug_path, &combined_threshold);
    }

    morphologyEx(&combined_threshold, MORPH_DILATE, kernel_large, 3);

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_08_dilated.png",
                 debug_prefix);
        save_image(debug_path, &combined_threshold);
    }

    morphologyEx(&combined_threshold, MORPH_ERODE, kernel_small, 2);

    if (debug_prefix)
    {
        char debug_path[256];
        snprintf(debug_path, sizeof(debug_path), "%s_09_eroded.png",
                 debug_prefix);
        save_image(debug_path, &combined_threshold);
    }

    freeStructuringElement(kernel_small);
    freeStructuringElement(kernel_large);

    Contours *contours = findContours(&combined_threshold, 1); // RETR_LIST
    if (!contours)
    {
        fprintf(stderr, "Error: Failed to find contours\n");
        free_image(&image);
        free_image(&combined_threshold);
        return NULL;
    }

    BoundingBoxArray *text_regions = malloc(sizeof(BoundingBoxArray));
    if (!text_regions)
    {
        freeContours(contours);
        free_image(&image);
        free_image(&combined_threshold);
        return NULL;
    }
    initBoundingBoxArray(text_regions);

    for (int i = 0; i < contours->count; i++)
    {
        Rect rect;
        if (boundingRect(&contours->contours[i], &rect))
        {
            int x = rect.x;
            int y = rect.y;
            int w = rect.width;
            int h = rect.height;

            float aspect_ratio = (float)w / h;
            int area = w * h;

            if (area < 100)
                continue;
            if (area >
                combined_threshold.width * combined_threshold.height * 0.15)
                continue;
            if (aspect_ratio < 1.5 || aspect_ratio > 15)
                continue;
            if (h > combined_threshold.height * 0.3)
                continue;
            if (w > combined_threshold.width * 0.4)
                continue;
            if (h < 10 || w < 15)
                continue;

            BoundingBox box = {x, y, w, h};
            addBoundingBox(text_regions, box);
        }
    }

    freeContours(contours);
    free_image(&combined_threshold);

    printf("Found %d filtered text regions\n", text_regions->count);

    BoundingBoxArray *merged_boxes = merge_nearby_boxes(text_regions, 20, 0.3);

    freeBoundingBoxArray(text_regions);
    free(text_regions);

    if (!merged_boxes)
    {
        free_image(&image);
        return NULL;
    }

    printf("Found %d merged boxes\n", merged_boxes->count);

    BoundingBoxArray *filtered_regions = malloc(sizeof(BoundingBoxArray));
    if (!filtered_regions)
    {
        freeBoundingBoxArray(merged_boxes);
        free(merged_boxes);
        free_image(&image);
        return NULL;
    }
    initBoundingBoxArray(filtered_regions);

    for (int i = 0; i < merged_boxes->count; i++)
    {
        BoundingBox box = merged_boxes->boxes[i];
        int area = box.width * box.height;

        if (area < 200 || box.height < 12 || box.width < 20)
            continue;
        if (area > image.width * image.height * 0.1 ||
            box.height > image.height * 0.25 || box.width > image.width * 0.35)
            continue;

        addBoundingBox(filtered_regions, box);
    }

    freeBoundingBoxArray(merged_boxes);
    free(merged_boxes);

    for (int i = 0; i < filtered_regions->count && i < 5; i++)
    {
        BoundingBox box = filtered_regions->boxes[i];
        printf("  Box %d: (%d, %d, %d, %d)\n", i, box.x, box.y, box.width,
               box.height);
    }

    WordGroups *groups = find_word_groups(filtered_regions, 150, 15);

    if (!groups)
    {
        freeBoundingBoxArray(filtered_regions);
        free(filtered_regions);
        free_image(&image);
        return NULL;
    }

    printf("Found %d word groups\n", groups->count);

    BoundingBoxArray *main_group = select_main_word_group(groups);

    if (main_group)
    {
        printf("Selected main group with %d words\n", main_group->count);
    }
    else
    {
        printf("No main word group selected\n");
    }

    freeWordGroups(groups);
    free(groups);
    freeBoundingBoxArray(filtered_regions);
    free(filtered_regions);
    free_image(&image);

    return main_group;
}
