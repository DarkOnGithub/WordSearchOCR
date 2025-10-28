#include "contour_analysis.h"
#include "../image/operations.h"
#include <stdlib.h>
#include <string.h>

Contour *find_largest_contour(const Contours *contours)
{
    if (!contours || contours->count == 0)
    {
        return NULL;
    }

    int max_area = 0;
    Contour *largest_contour = NULL;

    for (int i = 0; i < contours->count; i++)
    {
        Contour *contour = &contours->contours[i];
        Rect bounding_rect;

        if (boundingRect(contour, &bounding_rect))
        {
            int area = bounding_rect.width * bounding_rect.height;
            if (area > max_area)
            {
                max_area = area;
                largest_contour = contour;
            }
        }
    }

    return largest_contour;
}

int find_safe_line_position(int target_pos, int is_horizontal, int max_offset,
                            const Image *binary_img, int img_width,
                            int img_height)
{
    if (!binary_img || binary_img->gray_pixels == NULL)
    {
        return target_pos;
    }

    if (is_horizontal)
    {
        int start_y = target_pos - 2;
        int end_y = target_pos + 3;
        if (start_y < 0)
            start_y = 0;
        if (end_y > img_height)
            end_y = img_height;

        int check_height = end_y - start_y;
        if (check_height <= 0)
            return target_pos;

        int has_letters = 0;
        for (int y = start_y; y < end_y && !has_letters; y++)
        {
            for (int x = 0; x < img_width && !has_letters; x++)
            {
                int idx = y * img_width + x;
                if (binary_img->gray_pixels[idx] == 255)
                {
                    has_letters = 1;
                }
            }
        }

        if (has_letters)
        {
            for (int offset = 1; offset <= max_offset; offset++)
            {
                // Shift up
                int check_start_y = target_pos - offset - 2;
                int check_end_y = target_pos - offset + 3;
                if (check_start_y < 0)
                    check_start_y = 0;
                if (check_end_y > img_height)
                    check_end_y = img_height;

                if (check_end_y > check_start_y)
                {
                    int has_letters_up = 0;
                    for (int y = check_start_y;
                         y < check_end_y && !has_letters_up; y++)
                    {
                        for (int x = 0; x < img_width && !has_letters_up; x++)
                        {
                            int idx = y * img_width + x;
                            if (binary_img->gray_pixels[idx] == 255)
                            {
                                has_letters_up = 1;
                            }
                        }
                    }
                    if (!has_letters_up)
                    {
                        return target_pos - offset;
                    }
                }

                // Shift down
                check_start_y = target_pos + offset - 2;
                check_end_y = target_pos + offset + 3;
                if (check_start_y < 0)
                    check_start_y = 0;
                if (check_end_y > img_height)
                    check_end_y = img_height;

                if (check_end_y > check_start_y)
                {
                    int has_letters_down = 0;
                    for (int y = check_start_y;
                         y < check_end_y && !has_letters_down; y++)
                    {
                        for (int x = 0; x < img_width && !has_letters_down; x++)
                        {
                            int idx = y * img_width + x;
                            if (binary_img->gray_pixels[idx] == 255)
                            {
                                has_letters_down = 1;
                            }
                        }
                    }
                    if (!has_letters_down)
                    {
                        return target_pos + offset;
                    }
                }
            }
        }
    }
    else
    {
        int start_x = target_pos - 2;
        int end_x = target_pos + 3;
        if (start_x < 0)
            start_x = 0;
        if (end_x > img_width)
            end_x = img_width;

        int check_width = end_x - start_x;
        if (check_width <= 0)
            return target_pos;

        int has_letters = 0;
        for (int x = start_x; x < end_x && !has_letters; x++)
        {
            for (int y = 0; y < img_height && !has_letters; y++)
            {
                int idx = y * img_width + x;
                if (binary_img->gray_pixels[idx] == 255)
                {
                    has_letters = 1;
                }
            }
        }

        if (has_letters)
        {
            for (int offset = 1; offset <= max_offset; offset++)
            {
                // Shift left
                int check_start_x = target_pos - offset - 2;
                int check_end_x = target_pos - offset + 3;
                if (check_start_x < 0)
                    check_start_x = 0;
                if (check_end_x > img_width)
                    check_end_x = img_width;

                if (check_end_x > check_start_x)
                {
                    int has_letters_left = 0;
                    for (int x = check_start_x;
                         x < check_end_x && !has_letters_left; x++)
                    {
                        for (int y = 0; y < img_height && !has_letters_left;
                             y++)
                        {
                            int idx = y * img_width + x;
                            if (binary_img->gray_pixels[idx] == 255)
                            {
                                has_letters_left = 1;
                            }
                        }
                    }
                    if (!has_letters_left)
                    {
                        return target_pos - offset;
                    }
                }

                // Shift right
                check_start_x = target_pos + offset - 2;
                check_end_x = target_pos + offset + 3;
                if (check_start_x < 0)
                    check_start_x = 0;
                if (check_end_x > img_width)
                    check_end_x = img_width;

                if (check_end_x > check_start_x)
                {
                    int has_letters_right = 0;
                    for (int x = check_start_x;
                         x < check_end_x && !has_letters_right; x++)
                    {
                        for (int y = 0; y < img_height && !has_letters_right;
                             y++)
                        {
                            int idx = y * img_width + x;
                            if (binary_img->gray_pixels[idx] == 255)
                            {
                                has_letters_right = 1;
                            }
                        }
                    }
                    if (!has_letters_right)
                    {
                        return target_pos + offset;
                    }
                }
            }
        }
    }
    return target_pos; // Return original if no safe position found
}

Contours *filter_letter_contours(const Contours *contours)
{
    if (!contours)
    {
        return NULL;
    }

    // First pass: count valid contours
    int valid_count = 0;
    for (int i = 0; i < contours->count; i++)
    {
        Rect bounding_rect;
        if (!boundingRect(&contours->contours[i], &bounding_rect))
        {
            continue;
        }

        int area = bounding_rect.width * bounding_rect.height;
        double aspect_ratio =
            bounding_rect.width > 0
                ? (double)bounding_rect.height / bounding_rect.width
                : 0.0;

        if (area > 20 && area < 5000 && aspect_ratio > 0.3 &&
            aspect_ratio < 3.0 && bounding_rect.width > 8 &&
            bounding_rect.height > 8)
        {
            valid_count++;
        }
    }

    if (valid_count == 0)
    {
        Contours *result = (Contours *)malloc(sizeof(Contours));
        if (!result)
            return NULL;
        result->contours = NULL;
        result->count = 0;
        result->capacity = 0;
        return result;
    }

    Contours *result = (Contours *)malloc(sizeof(Contours));
    if (!result)
        return NULL;

    result->contours = (Contour *)malloc(sizeof(Contour) * valid_count);
    if (!result->contours)
    {
        free(result);
        return NULL;
    }
    result->capacity = valid_count;
    result->count = 0;

    for (int i = 0; i < contours->count; i++)
    {
        Rect bounding_rect;
        if (!boundingRect(&contours->contours[i], &bounding_rect))
        {
            continue;
        }

        int area = bounding_rect.width * bounding_rect.height;
        double aspect_ratio =
            bounding_rect.width > 0
                ? (double)bounding_rect.height / bounding_rect.width
                : 0.0;

        if (area > 20 && area < 5000 && aspect_ratio > 0.3 &&
            aspect_ratio < 3.0 && bounding_rect.width > 8 &&
            bounding_rect.height > 8)
        {

            Contour *src = &contours->contours[i];
            Contour *dst = &result->contours[result->count];

            dst->points = (Point *)malloc(sizeof(Point) * src->count);
            if (!dst->points)
            {
                freeContours(result);
                return NULL;
            }

            memcpy(dst->points, src->points, sizeof(Point) * src->count);
            dst->count = src->count;
            dst->capacity = src->count;
            result->count++;
        }
    }

    return result;
}

void translateContours(Contours *contours, int offset_x, int offset_y)
{
    if (!contours)
    {
        return;
    }

    for (int i = 0; i < contours->count; i++)
    {
        Contour *contour = &contours->contours[i];
        for (int j = 0; j < contour->count; j++)
        {
            contour->points[j].x += offset_x;
            contour->points[j].y += offset_y;
        }
    }
}
