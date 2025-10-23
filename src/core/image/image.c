#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

void load_image(const char* path, Image* image) {
    if (!path || !image) {
        fprintf(stderr, "Error: Invalid parameters to load_image\n");
        return;
    }

    memset(image, 0, sizeof(Image));

    SDL_Surface* loaded_surface = NULL;

    loaded_surface = IMG_Load(path);
    if (!loaded_surface) {
        fprintf(stderr, "Error loading image %s: %s\n", path, IMG_GetError());
        return;
    }
    SDL_Surface* optimized_surface = SDL_ConvertSurfaceFormat(loaded_surface,
                                                              SDL_PIXELFORMAT_RGBA32, 0);
    SDL_FreeSurface(loaded_surface);

    if (!optimized_surface) {
        fprintf(stderr, "Error optimizing surface: %s\n", SDL_GetError());
        return;
    }

    image->width = optimized_surface->w;
    image->height = optimized_surface->h;
    image->is_grayscale = false;

    size_t pixel_count = image->width * image->height;
    image->rgba_pixels = (uint32_t*)malloc(pixel_count * sizeof(uint32_t));
    if (!image->rgba_pixels) {
        fprintf(stderr, "Error: Failed to allocate pixel buffer\n");
        SDL_FreeSurface(optimized_surface);
        return;
    }

    uint32_t* src_pixels = (uint32_t*)optimized_surface->pixels;
    for (size_t i = 0; i < pixel_count; i++) {
        uint32_t abgr = src_pixels[i];
        uint8_t a = (abgr >> 24) & 0xFF;
        uint8_t b = (abgr >> 16) & 0xFF;
        uint8_t g = (abgr >> 8) & 0xFF;
        uint8_t r = abgr & 0xFF;
        image->rgba_pixels[i] = (r << 24) | (g << 16) | (b << 8) | a;
    }
    SDL_FreeSurface(optimized_surface);

    printf("Successfully loaded image: %s (%dx%d)\n", path, image->width, image->height);
}

void convert_to_grayscale(Image* image) {
    if (!image) {
        fprintf(stderr, "Error: Invalid image for grayscale conversion\n");
        return;
    }

    if (!image->rgba_pixels) {
        fprintf(stderr, "Error: Image has no RGBA pixel data to convert\n");
        return;
    }

    if (image->is_grayscale) {
        printf("Image is already grayscale\n");
        return;
    }

    size_t pixel_count = image->width * image->height;
    image->gray_pixels = (uint8_t*)malloc(pixel_count * sizeof(uint8_t));
    if (!image->gray_pixels) {
        fprintf(stderr, "Error: Failed to allocate grayscale pixel buffer\n");
        return;
    }
    uint32_t* rgba_pixels = image->rgba_pixels;
    for (int i = 0; i < pixel_count; i++) {
        uint32_t pixel = rgba_pixels[i];
        uint8_t r = (pixel >> 24) & 0xFF;
        uint8_t g = (pixel >> 16) & 0xFF;
        uint8_t b = (pixel >> 8) & 0xFF;

        image->gray_pixels[i] = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    }

    free(image->rgba_pixels);
    image->rgba_pixels = NULL;

    image->is_grayscale = true;
    printf("Converted image to grayscale (%dx%d)\n", image->width, image->height);
}

void save_image(const char* path, Image* image) {
    if (!path || !image) {
        fprintf(stderr, "Error: Invalid parameters to save_image\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels)) {
        fprintf(stderr, "Error: Image has no pixel data\n");
        return;
    }

    SDL_Surface* temp_surface = NULL;
    uint32_t* rgba_pixels = NULL;
    bool temp_pixels_allocated = false;

    if (image->is_grayscale) {
        size_t pixel_count = image->width * image->height;
        rgba_pixels = (uint32_t*)malloc(pixel_count * sizeof(uint32_t));
        if (!rgba_pixels) {
            fprintf(stderr, "Error: Failed to allocate temporary RGBA buffer for save\n");
            return;
        }

        for (int i = 0; i < pixel_count; i++) {
            uint8_t gray = image->gray_pixels[i];
            rgba_pixels[i] = (gray << 24) | (gray << 16) | (gray << 8) | 0xFF;
        }

        temp_pixels_allocated = true;
    } else {
        rgba_pixels = image->rgba_pixels;
    }

    temp_surface = SDL_CreateRGBSurfaceFrom(
        rgba_pixels,
        image->width,
        image->height,
        32,
        image->width * 4,
        0xFF000000,
        0x00FF0000,
        0x0000FF00,
        0x000000FF
    );

    if (!temp_surface) {
        fprintf(stderr, "Error creating surface for save: %s\n", SDL_GetError());
        if (temp_pixels_allocated) free(rgba_pixels);
        return;
    }

    int result = IMG_SavePNG(temp_surface, path);
    if (result != 0) {
        fprintf(stderr, "Error saving image %s: %s\n", path, IMG_GetError());
    } else {
        printf("Successfully saved image: %s\n", path);
    }

    SDL_FreeSurface(temp_surface);
    if (temp_pixels_allocated) free(rgba_pixels);
}

Tensor* to_tensor(Image* image) {
    if (!image) {
        fprintf(stderr, "Error: Invalid parameters to to_tensor\n");
        return NULL;
    }

    Tensor* tensor = tensor_create(1, image->is_grayscale ? 1 : 4, image->width, image->height);
    if (image->is_grayscale) {
        for (int i = 0; i < image->width * image->height; i++) {
            tensor->data[i] = image->gray_pixels[i] / 255.0f;
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

void cpy_image(const Image* image, Image* image_cpy) {
    if (!image || !image_cpy) {
        fprintf(stderr, "Error: Invalid parameters to cpy_image\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels)) {
        fprintf(stderr, "Error: Source image has no pixel data\n");
        return;
    }

    image_cpy->width = image->width;
    image_cpy->height = image->height;
    image_cpy->is_grayscale = image->is_grayscale;

    size_t pixel_count = image->width * image->height;

    if (image->is_grayscale) {
        image_cpy->gray_pixels = (uint8_t*)malloc(pixel_count * sizeof(uint8_t));
        if (!image_cpy->gray_pixels) {
            fprintf(stderr, "Error: Failed to allocate memory for grayscale copy\n");
            return;
        }
        memcpy(image_cpy->gray_pixels, image->gray_pixels, pixel_count * sizeof(uint8_t));
    } else {
        image_cpy->rgba_pixels = (uint32_t*)malloc(pixel_count * sizeof(uint32_t));
        if (!image_cpy->rgba_pixels) {
            fprintf(stderr, "Error: Failed to allocate memory for RGBA copy\n");
            return;
        }
        memcpy(image_cpy->rgba_pixels, image->rgba_pixels, pixel_count * sizeof(uint32_t));
    }
}

static void draw_horizontal_line(Image* image, int x1, int x2, int y, uint8_t gray_color, uint32_t rgba_color) {
    for (int x = x1; x < x2; x++) {
        if (image->is_grayscale) {
            image->gray_pixels[y * image->width + x] = gray_color;
        } else {
            image->rgba_pixels[y * image->width + x] = rgba_color;
        }
    }
}

static void draw_vertical_line(Image* image, int x, int y1, int y2, uint8_t gray_color, uint32_t rgba_color) {
    for (int y = y1; y < y2; y++) {
        if (image->is_grayscale) {
            image->gray_pixels[y * image->width + x] = gray_color;
        } else {
            image->rgba_pixels[y * image->width + x] = rgba_color;
        }
    }
}

void draw_rectangle(Image* image, int x, int y, int width, int height, bool fill, int thickness, uint32_t color) {
    if (!image) {
        fprintf(stderr, "Error: Invalid image for draw_rectangle\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels)) {
        fprintf(stderr, "Error: Image has no pixel data for draw_rectangle\n");
        return;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = (x + width > image->width) ? image->width : x + width;
    int end_y = (y + height > image->height) ? image->height : y + height;

    if (start_x >= end_x || start_y >= end_y) {
        return;
    }

    uint8_t gray_color = (color >> 24) & 0xFF;

    if (fill) {
        if (image->is_grayscale) {
            for (int i = start_x; i < end_x; i++) {
                for (int j = start_y; j < end_y; j++) {
                    image->gray_pixels[j * image->width + i] = gray_color;
                }
            }
        } else {
            for (int i = start_x; i < end_x; i++) {
                for (int j = start_y; j < end_y; j++) {
                    image->rgba_pixels[j * image->width + i] = color;
                }
            }
        }
    } else {
        int max_thickness = thickness;
        int max_possible_thickness = (end_x - start_x < end_y - start_y) ?
            (end_x - start_x) / 2 : (end_y - start_y) / 2;
        if (max_thickness > max_possible_thickness) {
            max_thickness = max_possible_thickness;
        }

        for (int t = 0; t < max_thickness; t++) {
            int inner_start_x = start_x + t;
            int inner_start_y = start_y + t;
            int inner_end_x = end_x - t;
            int inner_end_y = end_y - t;

            if (inner_start_x >= inner_end_x || inner_start_y >= inner_end_y) {
                break;
            }

            draw_horizontal_line(image, inner_start_x, inner_end_x, inner_start_y, gray_color, color);

            if (inner_end_y - 1 != inner_start_y) {
                draw_horizontal_line(image, inner_start_x, inner_end_x, inner_end_y - 1, gray_color, color);
            }

            if (inner_end_y - inner_start_y > 2) {
                draw_vertical_line(image, inner_start_x, inner_start_y + 1, inner_end_y - 1, gray_color, color);

                if (inner_end_x - 1 != inner_start_x) {
                    draw_vertical_line(image, inner_end_x - 1, inner_start_y + 1, inner_end_y - 1, gray_color, color);
                }
            }
        }
    }
}

void extract_rectangle(const Image* image, int x, int y, int width, int height, Image* extracted_image) {
    if (!image || !extracted_image) {
        fprintf(stderr, "Error: Invalid parameters to extract_rectangle\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels)) {
        fprintf(stderr, "Error: Source image has no pixel data\n");
        return;
    }

    if (width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid rectangle dimensions\n");
        return;
    }

    int start_x = x < 0 ? 0 : x;
    int start_y = y < 0 ? 0 : y;
    int end_x = (x + width > image->width) ? image->width : x + width;
    int end_y = (y + height > image->height) ? image->height : y + height;

    if (start_x >= end_x || start_y >= end_y) {
        fprintf(stderr, "Error: Rectangle is completely outside image bounds\n");
        return;
    }

    int extracted_width = end_x - start_x;
    int extracted_height = end_y - start_y;

    extracted_image->width = extracted_width;
    extracted_image->height = extracted_height;
    extracted_image->is_grayscale = image->is_grayscale;

    size_t pixel_count = extracted_width * extracted_height;

    if (image->is_grayscale) {
        extracted_image->gray_pixels = (uint8_t*)malloc(pixel_count * sizeof(uint8_t));
        if (!extracted_image->gray_pixels) {
            fprintf(stderr, "Error: Failed to allocate memory for extracted grayscale pixels\n");
            return;
        }

        for (int row = 0; row < extracted_height; row++) {
            int src_row = start_y + row;
            int src_start_idx = src_row * image->width + start_x;
            int dst_start_idx = row * extracted_width;
            memcpy(&extracted_image->gray_pixels[dst_start_idx],
                   &image->gray_pixels[src_start_idx],
                   extracted_width * sizeof(uint8_t));
        }
    } else {
        extracted_image->rgba_pixels = (uint32_t*)malloc(pixel_count * sizeof(uint32_t));
        if (!extracted_image->rgba_pixels) {
            fprintf(stderr, "Error: Failed to allocate memory for extracted RGBA pixels\n");
            return;
        }

        for (int row = 0; row < extracted_height; row++) {
            int src_row = start_y + row;
            int src_start_idx = src_row * image->width + start_x;
            int dst_start_idx = row * extracted_width;
            memcpy(&extracted_image->rgba_pixels[dst_start_idx],
                   &image->rgba_pixels[src_start_idx],
                   extracted_width * sizeof(uint32_t));
        }
    }

    printf("Successfully extracted rectangle (%dx%d) from image\n", extracted_width, extracted_height);
}

void free_image(Image* image) {
    if (!image) {
        return;
    }

    if (image->is_grayscale && image->gray_pixels) {
        free(image->gray_pixels);
        image->gray_pixels = NULL;
    } else if (!image->is_grayscale && image->rgba_pixels) {
        free(image->rgba_pixels);
        image->rgba_pixels = NULL;
    }

    image->width = 0;
    image->height = 0;
    image->is_grayscale = false;
}

