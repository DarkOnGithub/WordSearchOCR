#include "image/image.h"
#include "nn/core/tensor.h"
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gtk/gtk.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void load_image(const char *path, Image *image)
{
    if (!path || !image)
    {
        fprintf(stderr, "Error: Invalid parameters to load_image\n");
        return;
    }

    memset(image, 0, sizeof(Image));

    GError *error = NULL;
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file(path, &error);
    if (!pixbuf)
    {
        fprintf(stderr, "Error loading image %s: %s\n", path, error->message);
        g_error_free(error);
        return;
    }

    GdkPixbuf *rgba_pixbuf = NULL;
    if (gdk_pixbuf_get_has_alpha(pixbuf) &&
        gdk_pixbuf_get_n_channels(pixbuf) == 4)
    {
        rgba_pixbuf = pixbuf;
    }
    else
    {
        rgba_pixbuf = gdk_pixbuf_add_alpha(pixbuf, FALSE, 0, 0, 0);
        if (!rgba_pixbuf)
        {
            fprintf(stderr, "Error converting image to RGBA format\n");
            g_object_unref(pixbuf);
            return;
        }
        g_object_unref(pixbuf);
    }

    image->width = gdk_pixbuf_get_width(rgba_pixbuf);
    image->height = gdk_pixbuf_get_height(rgba_pixbuf);
    image->is_grayscale = false;

    size_t pixel_count = image->width * image->height;
    image->rgba_pixels = (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
    if (!image->rgba_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate pixel buffer\n");
        g_object_unref(rgba_pixbuf);
        return;
    }

    guchar *src_pixels = gdk_pixbuf_get_pixels(rgba_pixbuf);
    int rowstride = gdk_pixbuf_get_rowstride(rgba_pixbuf);

    for (int y = 0; y < image->height; y++)
    {
        for (int x = 0; x < image->width; x++)
        {
            int src_idx = y * rowstride + x * 4;
            uint8_t r = src_pixels[src_idx];
            uint8_t g = src_pixels[src_idx + 1];
            uint8_t b = src_pixels[src_idx + 2];
            uint8_t a = src_pixels[src_idx + 3];
            image->rgba_pixels[y * image->width + x] =
                (r << 24) | (g << 16) | (b << 8) | a;
        }
    }

    g_object_unref(rgba_pixbuf);

    printf("Successfully loaded image: %s (%dx%d)\n", path, image->width,
           image->height);
}

void convert_to_grayscale(Image *image)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for grayscale conversion\n");
        return;
    }

    if (!image->rgba_pixels)
    {
        fprintf(stderr, "Error: Image has no RGBA pixel data to convert\n");
        return;
    }

    if (image->is_grayscale)
    {
        printf("Image is already grayscale\n");
        return;
    }

    size_t pixel_count = image->width * image->height;
    image->gray_pixels = (uint8_t *)malloc(pixel_count * sizeof(uint8_t));
    if (!image->gray_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate grayscale pixel buffer\n");
        return;
    }
    uint32_t *rgba_pixels = image->rgba_pixels;
    for (int i = 0; i < pixel_count; i++)
    {
        uint32_t pixel = rgba_pixels[i];
        uint8_t r = (pixel >> 24) & 0xFF;
        uint8_t g = (pixel >> 16) & 0xFF;
        uint8_t b = (pixel >> 8) & 0xFF;

        image->gray_pixels[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }

    free(image->rgba_pixels);
    image->rgba_pixels = NULL;

    image->is_grayscale = true;
    printf("Converted image to grayscale (%dx%d)\n", image->width,
           image->height);
}

void save_image(const char *path, Image *image)
{
    if (!path || !image)
    {
        fprintf(stderr, "Error: Invalid parameters to save_image\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data\n");
        return;
    }

    GdkPixbuf *pixbuf = NULL;
    uint32_t *rgba_pixels = NULL;
    bool temp_pixels_allocated = false;

    if (image->is_grayscale)
    {
        size_t pixel_count = image->width * image->height;
        rgba_pixels = (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
        if (!rgba_pixels)
        {
            fprintf(
                stderr,
                "Error: Failed to allocate temporary RGBA buffer for save\n");
            return;
        }

        for (int i = 0; i < pixel_count; i++)
        {
            uint8_t gray = image->gray_pixels[i];
            rgba_pixels[i] = (gray << 24) | (gray << 16) | (gray << 8) | 0xFF;
        }

        temp_pixels_allocated = true;
    }
    else
    {
        rgba_pixels = image->rgba_pixels;
    }

    // Create pixel data in RGBA format for GdkPixbuf
    guchar *pixbuf_pixels = (guchar *)malloc(image->width * image->height * 4);
    if (!pixbuf_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate pixel buffer for pixbuf\n");
        if (temp_pixels_allocated)
            free(rgba_pixels);
        return;
    }

    for (int i = 0; i < image->width * image->height; i++)
    {
        uint32_t pixel = rgba_pixels[i];
        uint8_t r = (pixel >> 24) & 0xFF;
        uint8_t g = (pixel >> 16) & 0xFF;
        uint8_t b = (pixel >> 8) & 0xFF;
        uint8_t a = pixel & 0xFF;

        pixbuf_pixels[i * 4] = r;
        pixbuf_pixels[i * 4 + 1] = g;
        pixbuf_pixels[i * 4 + 2] = b;
        pixbuf_pixels[i * 4 + 3] = a;
    }

    pixbuf = gdk_pixbuf_new_from_data(pixbuf_pixels, GDK_COLORSPACE_RGB, TRUE,
                                      8, image->width, image->height,
                                      image->width * 4, NULL, NULL);

    if (!pixbuf)
    {
        fprintf(stderr, "Error creating pixbuf for save\n");
        free(pixbuf_pixels);
        if (temp_pixels_allocated)
            free(rgba_pixels);
        return;
    }

    GError *error = NULL;
    if (!gdk_pixbuf_savev(pixbuf, path, "png", NULL, NULL, &error))
    {
        fprintf(stderr, "Error saving image %s: %s\n", path, error->message);
        g_error_free(error);
    }
    else
    {
        printf("Successfully saved image: %s\n", path);
    }

    g_object_unref(pixbuf);
    free(pixbuf_pixels);
    if (temp_pixels_allocated)
        free(rgba_pixels);
}

/*
Tensor* to_tensor(Image* image) {
    if (!image) {
        fprintf(stderr, "Error: Invalid parameters to to_tensor\n");
        return NULL;
    }

    Tensor* tensor = tensor_create(1, image->is_grayscale ? 1 : 4, image->width,
image->height); if (image->is_grayscale) { for (int i = 0; i < image->width *
image->height; i++) { tensor->data[i] = image->gray_pixels[i] / 255.0f;
        }
    } else {
        int pixel_count = image->width * image->height;
        for (int i = 0; i < pixel_count; i++) {
            uint32_t rgba = image->rgba_pixels[i];
            uint8_t r = (rgba >> 24) & 0xFF;
            uint8_t g = (rgba >> 16) & 0xFF;
            uint8_t b = (rgba >> 8) & 0xFF;
            uint8_t a = rgba & 0xFF;

            tensor->data[i] = r / 255.0f;
            tensor->data[i + pixel_count] = g / 255.0f;
            tensor->data[i + 2 * pixel_count] = b / 255.0f;
            tensor->data[i + 3 * pixel_count] = a / 255.0f;
        }
    }
    return tensor;
}
*/

void cpy_image(const Image *image, Image *image_cpy)
{
    if (!image || !image_cpy)
    {
        fprintf(stderr, "Error: Invalid parameters to cpy_image\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Source image has no pixel data\n");
        return;
    }

    image_cpy->width = image->width;
    image_cpy->height = image->height;
    image_cpy->is_grayscale = image->is_grayscale;

    size_t pixel_count = image->width * image->height;

    if (image->is_grayscale)
    {
        image_cpy->gray_pixels =
            (uint8_t *)malloc(pixel_count * sizeof(uint8_t));
        if (!image_cpy->gray_pixels)
        {
            fprintf(stderr,
                    "Error: Failed to allocate memory for grayscale copy\n");
            return;
        }
        memcpy(image_cpy->gray_pixels, image->gray_pixels,
               pixel_count * sizeof(uint8_t));
    }
    else
    {
        image_cpy->rgba_pixels =
            (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
        if (!image_cpy->rgba_pixels)
        {
            fprintf(stderr, "Error: Failed to allocate memory for RGBA copy\n");
            return;
        }
        memcpy(image_cpy->rgba_pixels, image->rgba_pixels,
               pixel_count * sizeof(uint32_t));
    }
}

void gray_to_rgba(Image *image)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for gray_to_rgba\n");
        return;
    }

    if (!image->gray_pixels)
    {
        fprintf(stderr, "Error: Image has no grayscale pixel data to convert\n");
        return;
    }

    if (!image->is_grayscale)
    {
        printf("Image is already RGBA\n");
        return;
    }

    size_t pixel_count = image->width * image->height;
    image->rgba_pixels = (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
    if (!image->rgba_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate RGBA pixel buffer\n");
        return;
    }

    uint8_t *gray_pixels = image->gray_pixels;
    for (int i = 0; i < pixel_count; i++)
    {
        uint8_t gray_value = gray_pixels[i];
        image->rgba_pixels[i] = (gray_value << 24) | (gray_value << 16) |
                               (gray_value << 8) | 255;
    }

    free(image->gray_pixels);
    image->gray_pixels = NULL;

    image->is_grayscale = false;
    printf("Converted image to RGBA (%dx%d)\n", image->width, image->height);
}

static void draw_horizontal_line_static(Image *image, int x1, int x2, int y,
                                        uint32_t color)
{
    uint8_t gray_color = (color & 0xFF); // Extract gray value from color

    for (int x = x1; x < x2; x++)
    {
        if (image->is_grayscale)
        {
            image->gray_pixels[y * image->width + x] = gray_color;
        }
        else
        {
            image->rgba_pixels[y * image->width + x] = color;
        }
    }
}

static void draw_vertical_line_static(Image *image, int x, int y1, int y2,
                                      uint32_t color)
{
    uint8_t gray_color = (color & 0xFF); // Extract gray value from color

    for (int y = y1; y < y2; y++)
    {
        if (image->is_grayscale)
        {
            image->gray_pixels[y * image->width + x] = gray_color;
        }
        else
        {
            image->rgba_pixels[y * image->width + x] = color;
        }
    }
}

void draw_rectangle(Image *image, int x, int y, int width, int height,
                    bool fill, int thickness, uint32_t color)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for draw_rectangle\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data for draw_rectangle\n");
        return;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = (x + width > image->width) ? image->width : x + width;
    int end_y = (y + height > image->height) ? image->height : y + height;

    if (start_x >= end_x || start_y >= end_y)
    {
        return;
    }

    uint8_t gray_color = (color >> 24) & 0xFF;

    if (fill)
    {
        if (image->is_grayscale)
        {
            for (int i = start_x; i < end_x; i++)
            {
                for (int j = start_y; j < end_y; j++)
                {
                    image->gray_pixels[j * image->width + i] = gray_color;
                }
            }
        }
        else
        {
            for (int i = start_x; i < end_x; i++)
            {
                for (int j = start_y; j < end_y; j++)
                {
                    image->rgba_pixels[j * image->width + i] = color;
                }
            }
        }
    }
    else
    {
        int max_thickness = thickness;
        int max_possible_thickness = (end_x - start_x < end_y - start_y)
                                         ? (end_x - start_x) / 2
                                         : (end_y - start_y) / 2;
        if (max_thickness > max_possible_thickness)
        {
            max_thickness = max_possible_thickness;
        }

        for (int t = 0; t < max_thickness; t++)
        {
            int inner_start_x = start_x + t;
            int inner_start_y = start_y + t;
            int inner_end_x = end_x - t;
            int inner_end_y = end_y - t;

            if (inner_start_x >= inner_end_x || inner_start_y >= inner_end_y)
            {
                break;
            }

            draw_horizontal_line_static(image, inner_start_x, inner_end_x,
                                        inner_start_y, color);

            if (inner_end_y - 1 != inner_start_y)
            {
                draw_horizontal_line_static(image, inner_start_x, inner_end_x,
                                           inner_end_y - 1, color);
            }

            if (inner_end_y - inner_start_y > 2)
            {
                draw_vertical_line_static(image, inner_start_x, inner_start_y + 1,
                                         inner_end_y - 1, color);

                if (inner_end_x - 1 != inner_start_x)
                {
                    draw_vertical_line_static(image, inner_end_x - 1,
                                             inner_start_y + 1, inner_end_y - 1,
                                             color);
                }
            }
        }
    }
}

void draw_line(Image *image, int x1, int y1, int x2, int y2, uint32_t color)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for draw_line\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data for draw_line\n");
        return;
    }

    uint8_t gray_color = (color & 0xFF); // Extract gray value from color

    // Bresenham's line algorithm
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = x1 < x2 ? 1 : -1;
    int sy = y1 < y2 ? 1 : -1;
    int err = dx - dy;

    while (1)
    {
        if (x1 >= 0 && x1 < image->width && y1 >= 0 && y1 < image->height)
        {
            if (image->is_grayscale)
            {
                image->gray_pixels[y1 * image->width + x1] = gray_color;
            }
            else
            {
                image->rgba_pixels[y1 * image->width + x1] = color;
            }
        }

        if (x1 == x2 && y1 == y2)
            break;

        int e2 = 2 * err;
        if (e2 > -dy)
        {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx)
        {
            err += dx;
            y1 += sy;
        }
    }
}

void draw_horizontal_line(Image *image, int x1, int x2, int y, uint32_t color)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for draw_horizontal_line\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data for draw_horizontal_line\n");
        return;
    }

    uint8_t gray_color = (color & 0xFF); // Extract gray value from color

    if (x1 > x2)
    {
        int temp = x1;
        x1 = x2;
        x2 = temp;
    }

    for (int x = x1; x < x2; x++)
    {
        if (x >= 0 && x < image->width && y >= 0 && y < image->height)
        {
            if (image->is_grayscale)
            {
                image->gray_pixels[y * image->width + x] = gray_color;
            }
            else
            {
                image->rgba_pixels[y * image->width + x] = color;
            }
        }
    }
}

void draw_vertical_line(Image *image, int x, int y1, int y2, uint32_t color)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for draw_vertical_line\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data for draw_vertical_line\n");
        return;
    }

    uint8_t gray_color = (color & 0xFF); // Extract gray value from color

    if (y1 > y2)
    {
        int temp = y1;
        y1 = y2;
        y2 = temp;
    }

    for (int y = y1; y < y2; y++)
    {
        if (x >= 0 && x < image->width && y >= 0 && y < image->height)
        {
            if (image->is_grayscale)
            {
                image->gray_pixels[y * image->width + x] = gray_color;
            }
            else
            {
                image->rgba_pixels[y * image->width + x] = color;
            }
        }
    }
}

void extract_rectangle(const Image *image, int x, int y, int width, int height,
                       Image *extracted_image)
{
    if (!image || !extracted_image)
    {
        fprintf(stderr, "Error: Invalid parameters to extract_rectangle\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Source image has no pixel data\n");
        return;
    }

    if (width <= 0 || height <= 0)
    {
        fprintf(stderr, "Error: Invalid rectangle dimensions\n");
        return;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = (x + width > image->width) ? image->width : x + width;
    int end_y = (y + height > image->height) ? image->height : y + height;

    if (start_x >= end_x || start_y >= end_y)
    {
        fprintf(stderr,
                "Error: Rectangle is completely outside image bounds\n");
        return;
    }

    int extracted_width = end_x - start_x;
    int extracted_height = end_y - start_y;

    extracted_image->width = extracted_width;
    extracted_image->height = extracted_height;
    extracted_image->is_grayscale = image->is_grayscale;

    size_t pixel_count = extracted_width * extracted_height;

    if (image->is_grayscale)
    {
        extracted_image->gray_pixels =
            (uint8_t *)malloc(pixel_count * sizeof(uint8_t));
        if (!extracted_image->gray_pixels)
        {
            fprintf(stderr, "Error: Failed to allocate memory for extracted "
                            "grayscale pixels\n");
            return;
        }

        for (int row = 0; row < extracted_height; row++)
        {
            int src_row = start_y + row;
            int src_start_idx = src_row * image->width + start_x;
            int dst_start_idx = row * extracted_width;
            memcpy(&extracted_image->gray_pixels[dst_start_idx],
                   &image->gray_pixels[src_start_idx],
                   extracted_width * sizeof(uint8_t));
        }
    }
    else
    {
        extracted_image->rgba_pixels =
            (uint32_t *)malloc(pixel_count * sizeof(uint32_t));
        if (!extracted_image->rgba_pixels)
        {
            fprintf(
                stderr,
                "Error: Failed to allocate memory for extracted RGBA pixels\n");
            return;
        }

        for (int row = 0; row < extracted_height; row++)
        {
            int src_row = start_y + row;
            int src_start_idx = src_row * image->width + start_x;
            int dst_start_idx = row * extracted_width;
            memcpy(&extracted_image->rgba_pixels[dst_start_idx],
                   &image->rgba_pixels[src_start_idx],
                   extracted_width * sizeof(uint32_t));
        }
    }

    printf("Successfully extracted rectangle (%dx%d) from image\n",
           extracted_width, extracted_height);
}

void free_image(Image *image)
{
    if (!image)
    {
        return;
    }

    if (image->is_grayscale && image->gray_pixels)
    {
        free(image->gray_pixels);
        image->gray_pixels = NULL;
    }
    else if (!image->is_grayscale && image->rgba_pixels)
    {
        free(image->rgba_pixels);
        image->rgba_pixels = NULL;
    }

    image->width = 0;
    image->height = 0;
    image->is_grayscale = false;
}

Tensor* to_tensor(Image* image) {
    if (!image) {
        fprintf(stderr, "Error: Invalid parameters to to_tensor\n");
        return NULL;
    }

    int channels = image->is_grayscale ? 1 : 4;
    int shape[4] = {1, channels, image->height, image->width};
    Tensor* tensor = tensor_create(shape, 4);

    if (image->is_grayscale) {
        for (int i = 0; i < image->width * image->height; i++) {
            tensor->data[i] = image->gray_pixels[i] / 255.0f;
        }
    } else {
        for (int i = 0; i < image->width * image->height * 4; i++) {
            tensor->data[i] = image->rgba_pixels[i] / 255.0f;
        }
    }

    return tensor;
}

void resize_grayscale_image(Image* src, Image* dst, int target_width, int target_height) {
    if (!src || !dst || !src->is_grayscale || !src->gray_pixels) {
        fprintf(stderr, "Error: Invalid source image for resizing\n");
        return;
    }

    dst->width = target_width;
    dst->height = target_height;
    dst->is_grayscale = true;
    dst->rgba_pixels = NULL;
    dst->gray_pixels = (uint8_t*)malloc(target_width * target_height * sizeof(uint8_t));

    if (!dst->gray_pixels) {
        fprintf(stderr, "Error: Failed to allocate memory for resized image\n");
        return;
    }

    float x_ratio = (float)src->width / target_width;
    float y_ratio = (float)src->height / target_height;

    for (int y = 0; y < target_height; y++) {
        for (int x = 0; x < target_width; x++) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

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

            dst->gray_pixels[y * target_width + x] = (uint8_t)roundf(interpolated);
        }
    }
}

void draw_angled_capsule(Image *image, int center_x, int center_y, int length, int width, double angle, int radius, uint32_t color)
{
    if (!image || length <= 0 || width <= 0 || radius < 0)
    {
        return;
    }

    // Allow radius to be larger than half the width for more prominent circular ends
    // No upper limit restriction for capsule appearance

    // Convert angle to radians
    double rad_angle = angle * M_PI / 180.0;
    double cos_angle = cos(rad_angle);
    double sin_angle = sin(rad_angle);

    // Calculate the half dimensions
    double half_length = length / 2.0;
    double half_width = width / 2.0;

    // Calculate bounding box of the rotated capsule
    double corners[4][2] = {
        {-half_length, -half_width},
        {half_length, -half_width},
        {-half_length, half_width},
        {half_length, half_width}
    };

    int min_x = INT_MAX, max_x = INT_MIN;
    int min_y = INT_MAX, max_y = INT_MIN;

    for (int i = 0; i < 4; i++)
    {
        // Rotate the corner
        double rot_x = corners[i][0] * cos_angle - corners[i][1] * sin_angle;
        double rot_y = corners[i][0] * sin_angle + corners[i][1] * cos_angle;

        // Translate to center
        int px = (int)(center_x + rot_x);
        int py = (int)(center_y + rot_y);

        if (px < min_x) min_x = px;
        if (px > max_x) max_x = px;
        if (py < min_y) min_y = py;
        if (py > max_y) max_y = py;
    }

    // Add some padding for the rounded ends
    min_x -= radius;
    min_y -= radius;
    max_x += radius;
    max_y += radius;

    // Clamp to image bounds
    if (min_x < 0) min_x = 0;
    if (min_y < 0) min_y = 0;
    if (max_x >= image->width) max_x = image->width - 1;
    if (max_y >= image->height) max_y = image->height - 1;

    // Draw the capsule by checking each pixel in the bounding box
    for (int y = min_y; y <= max_y; y++)
    {
        for (int x = min_x; x <= max_x; x++)
        {
            // Transform pixel back to capsule's local coordinate system
            double dx = x - center_x;
            double dy = y - center_y;

            // Rotate back by negative angle
            double local_x = dx * cos_angle + dy * sin_angle;
            double local_y = -dx * sin_angle + dy * cos_angle;

            // Check if point is inside the capsule
            int inside = 0;

            // Check if in the rectangular part
            if (fabs(local_x) <= half_length - radius && fabs(local_y) <= half_width)
            {
                inside = 1;
            }
            // Check if in the left semicircle
            else if (local_x < -half_length + radius && fabs(local_x + half_length - radius) <= radius)
            {
                double dist_sq = (local_x + half_length - radius) * (local_x + half_length - radius) +
                                local_y * local_y;
                if (dist_sq <= radius * radius)
                {
                    inside = 1;
                }
            }
            // Check if in the right semicircle
            else if (local_x > half_length - radius && fabs(local_x - half_length + radius) <= radius)
            {
                double dist_sq = (local_x - half_length + radius) * (local_x - half_length + radius) +
                                local_y * local_y;
                if (dist_sq <= radius * radius)
                {
                    inside = 1;
                }
            }

            if (inside)
            {
                if (image->is_grayscale)
                {
                    image->gray_pixels[y * image->width + x] = (color >> 24) & 0xFF;
                }
                else if (image->rgba_pixels)
                {
                    // Alpha blending: blend capsule color with background
                    uint32_t bg_pixel = image->rgba_pixels[y * image->width + x];

                    // Extract components (ARGB format)
                    uint8_t bg_r = (bg_pixel >> 16) & 0xFF;
                    uint8_t bg_g = (bg_pixel >> 8) & 0xFF;
                    uint8_t bg_b = bg_pixel & 0xFF;

                    uint8_t fg_a = (color >> 24) & 0xFF;
                    uint8_t fg_r = (color >> 16) & 0xFF;
                    uint8_t fg_g = (color >> 8) & 0xFF;
                    uint8_t fg_b = color & 0xFF;

                    // Alpha blending formula
                    float alpha = fg_a / 255.0f;
                    uint8_t r = (uint8_t)(fg_r * alpha + bg_r * (1.0f - alpha));
                    uint8_t g = (uint8_t)(fg_g * alpha + bg_g * (1.0f - alpha));
                    uint8_t b = (uint8_t)(fg_b * alpha + bg_b * (1.0f - alpha));
                    uint8_t a = 0xFF; // Final pixel is always opaque

                    image->rgba_pixels[y * image->width + x] = (a << 24) | (r << 16) | (g << 8) | b;
                }
            }
        }
    }
}

void draw_rounded_rectangle(Image *image, int x, int y, int width, int height, int radius, uint32_t color)
{
    if (!image || width <= 0 || height <= 0 || radius < 0)
    {
        return;
    }

    // Ensure radius doesn't exceed half the smallest dimension
    int max_radius = (width < height ? width : height) / 2;
    if (radius > max_radius)
    {
        radius = max_radius;
    }

    // Draw the main rectangle (excluding corners)
    draw_rectangle(image, x + radius, y, width - 2 * radius, height, true, 0, color);
    draw_rectangle(image, x, y + radius, width, height - 2 * radius, true, 0, color);

    // Draw the four corner quarter-circles
    // Top-left corner
    for (int dy = 0; dy <= radius; dy++)
    {
        for (int dx = 0; dx <= radius; dx++)
        {
            if (dx * dx + dy * dy <= radius * radius)
            {
                int px = x + radius - dx;
                int py = y + radius - dy;
                if (px >= 0 && px < image->width && py >= 0 && py < image->height)
                {
                    if (image->is_grayscale)
                    {
                        image->gray_pixels[py * image->width + px] = (color >> 24) & 0xFF;
                    }
                    else if (image->rgba_pixels)
                    {
                        image->rgba_pixels[py * image->width + px] = color;
                    }
                }
            }
        }
    }

    // Top-right corner
    for (int dy = 0; dy <= radius; dy++)
    {
        for (int dx = 0; dx <= radius; dx++)
        {
            if (dx * dx + dy * dy <= radius * radius)
            {
                int px = x + width - radius + dx;
                int py = y + radius - dy;
                if (px >= 0 && px < image->width && py >= 0 && py < image->height)
                {
                    if (image->is_grayscale)
                    {
                        image->gray_pixels[py * image->width + px] = (color >> 24) & 0xFF;
                    }
                    else if (image->rgba_pixels)
                    {
                        image->rgba_pixels[py * image->width + px] = color;
                    }
                }
            }
        }
    }

    // Bottom-left corner
    for (int dy = 0; dy <= radius; dy++)
    {
        for (int dx = 0; dx <= radius; dx++)
        {
            if (dx * dx + dy * dy <= radius * radius)
            {
                int px = x + radius - dx;
                int py = y + height - radius + dy;
                if (px >= 0 && px < image->width && py >= 0 && py < image->height)
                {
                    if (image->is_grayscale)
                    {
                        image->gray_pixels[py * image->width + px] = (color >> 24) & 0xFF;
                    }
                    else if (image->rgba_pixels)
                    {
                        image->rgba_pixels[py * image->width + px] = color;
                    }
                }
            }
        }
    }

    // Bottom-right corner
    for (int dy = 0; dy <= radius; dy++)
    {
        for (int dx = 0; dx <= radius; dx++)
        {
            if (dx * dx + dy * dy <= radius * radius)
            {
                int px = x + width - radius + dx;
                int py = y + height - radius + dy;
                if (px >= 0 && px < image->width && py >= 0 && py < image->height)
                {
                    if (image->is_grayscale)
                    {
                        image->gray_pixels[py * image->width + px] = (color >> 24) & 0xFF;
                    }
                    else if (image->rgba_pixels)
                    {
                        image->rgba_pixels[py * image->width + px] = color;
                    }
                }
            }
        }
    }
}