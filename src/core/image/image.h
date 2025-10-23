#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>
#include <string.h>

struct Contour;

typedef struct Image {
    int width;
    int height;
    bool is_grayscale;
    uint32_t* rgba_pixels;
    uint8_t* gray_pixels;
} Image;

void load_image(const char* path, Image* image);
void convert_to_grayscale(Image* image);
void save_image(const char* path, Image* image);
void cpy_image(const Image* image, Image* image_cpy);
void draw_rectangle(Image* image, int x, int y, int width, int height, bool fill, int thickness, uint32_t color);
void extract_rectangle(const Image* image, int x, int y, int width, int height, Image* extracted_image);
//void to_tensor(Image* image, Tensor* tensor);
void free_image(Image* image);
