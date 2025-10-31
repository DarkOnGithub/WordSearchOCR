#include "gui/gui.h"
#include "wordsearch/processor.h"
#include "image/operations.h"
#include "image/image.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>


int main(int argc, char *argv[])
{
    if (argc >= 2)
    {
        if(strcmp(argv[1], "-r") == 0)
        {
            if(argc != 4)
            {
                printf("Usage: %s -r <image_path> <angle>\n", argv[0]);
                return 1;
            }
            char *image_path = argv[2];
            double angle = atof(argv[3]);
            Image image;
            load_image(image_path, &image);
            rotate_image(&image, angle);
            char output_path[256];
            char *dot_pos = strrchr(image_path, '.');
            if (dot_pos != NULL) {
                size_t base_len = dot_pos - image_path;
                strncpy(output_path, image_path, base_len);
                output_path[base_len] = '\0';
                strcat(output_path, "_rotated");
                strcat(output_path, dot_pos);
            } else {
                strcpy(output_path, image_path);
                strcat(output_path, "_rotated");
            }
            save_image(output_path, &image);
            free_image(&image);
            return 0;
        }
        else
        {
            char *image_path = argv[1];
            process_wordsearch_image(image_path, NULL);
            process_word_detection(image_path, NULL);
            return 0;
        }
    }
    else
    {
        // GUI mode
        return main_gui(argc, argv);
    }
}