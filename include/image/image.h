#pragma once

#include <gdk-pixbuf/gdk-pixbuf.h>
#include <gtk/gtk.h>
#include <stdbool.h>
#include <stdint.h>

#include "nn/core/tensor.h"

struct Contour;

typedef struct Image
{
    int width;
    int height;
    bool is_grayscale;
    uint32_t *rgba_pixels;
    uint8_t *gray_pixels;
} Image;

void load_image(const char *path, Image *image);
void convert_to_grayscale(Image *image);
void save_image(const char *path, Image *image);
void cpy_image(const Image *image, Image *image_cpy);
void gray_to_rgba(Image *image);
void draw_rectangle(Image *image, int x, int y, int width, int height,
                    bool fill, int thickness, uint32_t color);
void draw_line(Image *image, int x1, int y1, int x2, int y2, uint32_t color);
void draw_horizontal_line(Image *image, int x1, int x2, int y, uint32_t color);
void draw_vertical_line(Image *image, int x, int y1, int y2, uint32_t color);

/*
    Draw a rounded rectangle (capsule-like) on an image.
    @param image The image to draw on
    @param x X coordinate of the rectangle
    @param y Y coordinate of the rectangle
    @param width Width of the rectangle
    @param height Height of the rectangle
    @param radius Corner radius for rounded corners
    @param color Color to use (RGBA format)
*/
void draw_rounded_rectangle(Image *image, int x, int y, int width, int height, int radius, uint32_t color);

/*
    Draw an angled capsule (rotated rounded rectangle) on an image.
    @param image The image to draw on
    @param center_x X coordinate of the capsule center
    @param center_y Y coordinate of the capsule center
    @param length Length of the capsule (along its direction)
    @param width Width of the capsule (perpendicular to its direction)
    @param angle Rotation angle in degrees (0 = right, 90 = down, etc.)
    @param radius Corner radius for rounded ends
    @param color Color to use (RGBA format)
*/
void draw_angled_capsule(Image *image, int center_x, int center_y, int length, int width, double angle, int radius, uint32_t color);

/*
    Draw a filled rectangle with alpha blending on an image.
    @param image The image to draw on
    @param x X coordinate of the rectangle
    @param y Y coordinate of the rectangle
    @param width Width of the rectangle
    @param height Height of the rectangle
    @param color Color to use (RGBA format with alpha channel)
*/
void draw_filled_rectangle_alpha(Image *image, int x, int y, int width, int height, uint32_t color);
void extract_rectangle(const Image *image, int x, int y, int width, int height,
                       Image *extracted_image);
void resize_grayscale_image(Image* src, Image* dst, int target_width, int target_height);

Tensor* to_tensor(Image* image);
void free_image(Image *image);