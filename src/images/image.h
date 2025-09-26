#pragma once

#include <stdbool.h>
#include <SDL2/SDL.h>  // Uncomment when SDL2 is installed
#include <string.h>

typedef struct Image {
    SDL_Surface* surface;
    int width;
    int height;
    bool is_grayscale;
    void* pixels;
} Image;

void load_image(const char* path, Image* image);
void convert_to_grayscale(Image* image);
void save_image(const char* path, Image* image);
//void to_tensor(Image* image, Tensor* tensor);
void free_image(Image* image);
