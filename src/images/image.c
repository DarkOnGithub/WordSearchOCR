#include "image.h"
#include <stdio.h>
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
    image->surface = optimized_surface;
    image->width = optimized_surface->w;
    image->height = optimized_surface->h;
    image->is_grayscale = false;
    image->pixels = optimized_surface->pixels;

    printf("Successfully loaded image: %s (%dx%d)\n", path, image->width, image->height);
}

void convert_to_grayscale(Image* image) {
    if (!image || !image->surface || !image->pixels) {
        fprintf(stderr, "Error: Invalid image for grayscale conversion\n");
        return;
    }

    if (image->is_grayscale) {
        printf("Image is already grayscale\n");
        return;
    }

    SDL_LockSurface(image->surface);
    
    Uint32* pixels = (Uint32*)image->pixels;
    SDL_PixelFormat* format = image->surface->format;
    
    for (int y = 0; y < image->height; y++) {
        for (int x = 0; x < image->width; x++) {
            int pixel_index = y * image->width + x;
            Uint32 pixel = pixels[pixel_index];
            Uint8 r, g, b, a;
            SDL_GetRGBA(pixel, format, &r, &g, &b, &a);
            Uint8 gray = (Uint8)(0.299 * r + 0.587 * g + 0.114 * b);
            Uint32 gray_pixel = SDL_MapRGBA(format, gray, gray, gray, a);
            pixels[pixel_index] = gray_pixel;
        }
    }
    
    SDL_UnlockSurface(image->surface);
    image->is_grayscale = true;
    
    printf("Converted image to grayscale (%dx%d)\n", image->width, image->height);
}

void save_image(const char* path, Image* image) {
    if (!path || !image || !image->surface) {
        fprintf(stderr, "Error: Invalid parameters to save_image\n");
        return;
    }
    int result = -1;
    result = IMG_SavePNG(image->surface, path);
    if (result != 0) {
        fprintf(stderr, "Error saving image %s: %s\n", path, IMG_GetError());
        return;
    }   

    printf("Successfully saved image: %s\n", path);
}

void free_image(Image* image) {
    if (!image) {
        return;
    }

    if (image->surface) {
        SDL_FreeSurface(image->surface);
        image->surface = NULL;
    }

    image->width = 0;
    image->height = 0;
    image->is_grayscale = false;
    image->pixels = NULL;

    printf("Freed image memory\n");
}
