#pragma once

#include "nn/core/tensor.h"
#include "nn/cnn.h"

typedef struct {
    int height;
    int width;
    Tensor* grid;
} Grid;

typedef struct {
    Tensor* probabilities;
    char* image_name;
} Word;

typedef struct {
    Word* words;
    int count;
    int capacity;
} WordsArray;

typedef struct {
    int row, col;  // Position in grid
} Position;

typedef struct {
    Position* positions;  // Array of positions forming the word path
    int length;          // Number of positions
} Path;

typedef struct {
    char* word_str;      // The word that was found
    Position start_pos;  // Starting position
    char direction[20];  // Direction name ("Right", "DownRight", etc.)
    float log_prob_score; // Log-probability score
    Path path;           // The path of positions
} WordMatch;

Grid* create_grid(int height, int width, const char* letters_path, CNN* model);
void FreeGrid(Grid* grid);
char* grid_to_string(const Grid* grid);

WordsArray* create_words_array(const char* words_path, CNN* model);
void FreeWordsArray(WordsArray* words_array);
char* word_to_string(const Word* word);

// Probabilistic word search functions
int char_to_index(char c);
char index_to_char(int idx);
Tensor* create_log_prob_tensor(Tensor* prob_tensor);
WordMatch* find_best_word_match(const Grid* grid, const Word* word, const char* word_str);
void free_word_match(WordMatch* match);