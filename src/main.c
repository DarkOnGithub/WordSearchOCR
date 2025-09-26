#include "images/image.h"
#include <SDL2/SDL.h>  // Uncomment when SDL2 is installed

int main(int argc, char* argv[]){
    (void)argc;
    (void)argv;
    Image image;
    load_image("images/level_1_image_1.png", &image);
    convert_to_grayscale(&image);
    save_image("level_1_image_1_grayscale.png", &image);
    free_image(&image);
    return 0;
}