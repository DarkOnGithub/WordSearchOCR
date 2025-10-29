#include "../../include/image/image.h"
#include "../../include/wordsearch/word_detection.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void extract_word_images(const char *image_path, const BoundingBoxArray *words)
{
    if (!words || words->count == 0)
    {
        return;
    }

    Image original_image = {0};
    load_image(image_path, &original_image);

    char base_name[256];
    const char *slash = strrchr(image_path, '/');
    if (slash)
    {
        strcpy(base_name, slash + 1);
    }
    else
    {
        strcpy(base_name, image_path);
    }

    char *dot = strrchr(base_name, '.');
    if (dot)
    {
        *dot = '\0';
    }

    printf("Extracting %d word images from %s\n", words->count, image_path);

    for (int i = 0; i < words->count; i++)
    {
        BoundingBox box = words->boxes[i];

        int x = box.x;
        int y = box.y;
        int width = box.width;
        int height = box.height;

        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x + width > original_image.width)
            width = original_image.width - x;
        if (y + height > original_image.height)
            height = original_image.height - y;

        if (width <= 0 || height <= 0)
        {
            printf("Skipping invalid word bounding box: (%d, %d, %d, %d)\n",
                   box.x, box.y, box.width, box.height);
            continue;
        }

        Image word_image = {0};
        extract_rectangle(&original_image, x, y, width, height, &word_image);

        char word_filename[512];
        snprintf(word_filename, sizeof(word_filename), "words/%s_word_%d.png",
                 base_name, i + 1);
        save_image(word_filename, &word_image);
        printf("Saved word %d: %s\n", i + 1, word_filename);

        free_image(&word_image);
    }

    free_image(&original_image);
}

int process_image(const char *image_path)
{
    char debug_prefix[256];
    snprintf(debug_prefix, sizeof(debug_prefix), "debug_%s", image_path);

    char *dot = strrchr(debug_prefix, '.');
    if (dot)
    {
        *dot = '\0';
    }

    printf("Detecting words in image: %s\n", image_path);

    BoundingBoxArray *main_group = detect_words(image_path, debug_prefix);

    if (!main_group)
    {
        fprintf(stderr, "Failed to detect words in image\n");
        return 0;
    }

    printf("Found main word group with %d words\n", main_group->count);

    Image image;
    load_image(image_path, &image);

    if (!image.rgba_pixels && !image.gray_pixels)
    {
        fprintf(stderr, "Error: Could not load image for drawing\n");
        freeBoundingBoxArray(main_group);
        free(main_group);
        return 0;
    }

    if (image.is_grayscale)
    {
        printf("Skipping rectangle drawing on grayscale image\n");
    }
    else
    {
        draw_bounding_boxes(&image, main_group, 0xFF00FF00, 2);

        if (main_group->count > 1)
        {
            int min_x = INT_MAX, min_y = INT_MAX;
            int max_x = 0, max_y = 0;

            for (int i = 0; i < main_group->count; i++)
            {
                BoundingBox box = main_group->boxes[i];
                if (box.x < min_x)
                    min_x = box.x;
                if (box.y < min_y)
                    min_y = box.y;
                if (box.x + box.width > max_x)
                    max_x = box.x + box.width;
                if (box.y + box.height > max_y)
                    max_y = box.y + box.height;
            }

            draw_rectangle(&image, min_x, min_y, max_x - min_x, max_y - min_y,
                           false, 3, 0xFF0000FF);
        }

        char output_path[256];
        char base_name[256];
        const char *slash = strrchr(image_path, '/');
        if (slash)
        {
            strcpy(base_name, slash + 1);
        }
        else
        {
            strcpy(base_name, image_path);
        }

        dot = strrchr(base_name, '.');
        if (dot)
        {
            *dot = '\0';
        }

        snprintf(output_path, sizeof(output_path), "words_with_squares_%s.png",
                 base_name);
        save_image(output_path, &image);
        printf("Image saved as %s\n", output_path);
    }

    free_image(&image);

    extract_word_images(image_path, main_group);

    int word_count = main_group->count;
    freeBoundingBoxArray(main_group);
    free(main_group);

    printf("Total words detected: %d\n", word_count);
    return word_count;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        const char *test_images[] = {
            "images/level_3_image_2.png", "images/level_1_image_1.png",
            "images/level_2_image_1_rotated_25.0deg.png", NULL};

        printf("Running word detection on default test images...\n");

        for (int i = 0; test_images[i] != NULL; i++)
        {
            int word_count = process_image(test_images[i]);
            printf("Processed %s: %d words\n\n", test_images[i], word_count);
        }

        return 0;
    }

    for (int i = 1; i < argc; i++)
    {
        const char *image_path = argv[i];
        int word_count = process_image(image_path);
        if (i < argc - 1)
        {
            printf("\n");
        }
    }

    return 0;
}
