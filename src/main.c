#include "gui/gui.h"
#include "nn/core/tensor.h"
#include "wordsearch/processor.h"
#include "image/operations.h"
#include "image/image.h"
#include "processing/preprocessing.h"
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "nn/cnn.h"
#include "solver/solver.h"
#include "nn/inference.h"
#include "wordsearch/word_detection.h"

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
        }else if(strcmp(argv[1], "-s") == 0)
        {
            if(argc < 4)
            {
                printf("Usage: %s -s <words_path> <cells_path>\n", argv[0]);
                return 1;
            }
            char *words_path = argv[2];
            char *cells_path = argv[3];
            CNN* model = cnn_create();
            cnn_load_weights(model, 12);
            cnn_eval(model);
            Grid* grid = create_grid(15, 15, cells_path, model);
            char* grid_str = grid_to_string(grid);
            WordsArray* words_array = create_words_array(words_path, model);
            printf("Grid:\n%s\n", grid_str);
            free(grid_str);

            for(int i = 0; i < words_array->count; i++){
                Word* word = &words_array->words[i];
                char* word_str = word_to_string(word);
                if (word_str) {
                    WordMatch* word_match = find_best_word_match(grid, word, word_str);
                    if (word_match) {
                        printf("Found word '%s' at (%d,%d) going %s (score: %.3f)\n",
                               word_match->word_str,
                               word_match->start_pos.row,
                               word_match->start_pos.col,
                               word_match->direction,
                               word_match->log_prob_score);
                        free_word_match(word_match);
                    }
                    free(word_str);
                }
            }
            cnn_free(model);
            return 0;
        }
        else
        {
            char *image_path = argv[1];
            int num_rows, num_cols, num_cells;
            Rect *cell_bounding_boxes;
            int crop_offset_x = 0, crop_offset_y = 0;
            process_wordsearch_image(image_path, NULL, &num_rows, &num_cols, &cell_bounding_boxes, &num_cells,
                                   &crop_offset_x, &crop_offset_y);

            int text_region_offset_x = crop_offset_x;
            int text_region_offset_y = crop_offset_y;
            process_word_detection(image_path, NULL);
            CNN* model = cnn_create();
            cnn_load_weights(model, 12);
            cnn_eval(model);
            Grid* grid = create_grid(num_rows, num_cols, "cells", model);
            char* grid_str = grid_to_string(grid);
            WordsArray* words_array = create_words_array("words", model);
            printf("Grid:\n%s\n", grid_str);
            free(grid_str);

            // Collect all word matches
            WordMatch** word_matches = (WordMatch**)malloc(sizeof(WordMatch*) * words_array->count);
            int num_matches = 0;

            for(int i = 0; i < words_array->count; i++){
                Word* word = &words_array->words[i];
                char* word_str = word_to_string(word);
                if (word_str) {
                    WordMatch* word_match = find_best_word_match(grid, word, word_str);
                    if (word_match) {
                        printf("Found word '%s' at (%d,%d) going %s (score: %.3f)\n",
                               word_match->word_str,
                               word_match->start_pos.row,
                               word_match->start_pos.col,
                               word_match->direction,
                               word_match->log_prob_score);
                        word_matches[num_matches++] = word_match;
                    }
                    free(word_str);
                }
            }

            // Draw capsules around solved words if any were found
            if (num_matches > 0) {
                char output_path[256];
                char *basename = strrchr(image_path, '/');
                if (basename == NULL) {
                    basename = strrchr(image_path, '\\'); // Handle Windows paths
                }
                if (basename == NULL) {
                    basename = image_path; // No path separator found
                } else {
                    basename++; // Skip the separator
                }
                char *dot_pos = strrchr(basename, '.');
                if (dot_pos != NULL) {
                    sprintf(output_path, "solved/%.*s_solved%s", (int)(dot_pos - basename), basename, dot_pos);
                } else {
                    sprintf(output_path, "solved/%s_solved", basename);
                }

                draw_solved_words(image_path, word_matches, num_matches, num_rows, num_cols, cell_bounding_boxes,
                                  text_region_offset_x, text_region_offset_y, output_path);

                // Free word matches
                for (int i = 0; i < num_matches; i++) {
                    free_word_match(word_matches[i]);
                }
            }

            free(word_matches);
            if (cell_bounding_boxes) free(cell_bounding_boxes);
            return 0;
        }
    }
    else
    {
        // GUI mode
        return main_gui(argc, argv);
    }

    return 0;
}