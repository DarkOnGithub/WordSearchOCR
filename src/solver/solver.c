#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../include/solver/solver.h"
#include "../include/nn/layers/cross_entropy_loss.h"
#include "../include/image/image.h"
#include "../include/wordsearch/word_detection.h"

Grid* create_grid(int height, int width, const char* letters_path, CNN* model) {
    if (!letters_path || !model || height <= 0 || width <= 0) {
        fprintf(stderr, "Error: Invalid parameters to create_grid\n");
        return NULL;
    }

    int grid_shape[] = {height, width, 26};
    Tensor* grid_tensor = tensor_create_zero(grid_shape, 3);
    if (!grid_tensor) {
        fprintf(stderr, "Error: Failed to create grid tensor\n");
        return NULL;
    }

    DIR* dir = opendir(letters_path);
    if (!dir) {
        fprintf(stderr, "Error: Failed to open letters directory: %s\n", letters_path);
        tensor_free(grid_tensor);
        return NULL;
    }

    typedef struct {
        char filepath[1024];
        int row, col;
    } ImageInfo;

    ImageInfo* image_list = NULL;
    int image_count = 0;
    int image_capacity = 0;

    struct dirent* entry;

    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "cell_", 5) == 0 && strstr(entry->d_name, ".png")) {
            int row, col;
            if (sscanf(entry->d_name, "cell_%d_%d.png", &row, &col) == 2) {
                if (row >= 0 && row < height && col >= 0 && col < width) {
                    if (image_count >= image_capacity) {
                        image_capacity = image_capacity == 0 ? 64 : image_capacity * 2;
                        image_list = (ImageInfo*)realloc(image_list, sizeof(ImageInfo) * image_capacity);
                        if (!image_list) {
                            fprintf(stderr, "Error: Failed to allocate memory for image list\n");
                            closedir(dir);
                            tensor_free(grid_tensor);
                            return NULL;
                        }
                    }

                    ImageInfo* info = &image_list[image_count++];
                    snprintf(info->filepath, sizeof(info->filepath), "%s/%s", letters_path, entry->d_name);
                    info->row = row;
                    info->col = col;
                }
            }
        }
    }

    rewinddir(dir);

    closedir(dir);

    if (image_count == 0) {
        fprintf(stderr, "Warning: No valid cell images found\n");
        free(image_list);
        Grid* grid = (Grid*)malloc(sizeof(Grid));
        if (!grid) {
            fprintf(stderr, "Error: Failed to allocate Grid structure\n");
            tensor_free(grid_tensor);
            return NULL;
        }
        grid->height = height;
        grid->width = width;
        grid->grid = grid_tensor;
        return grid;
    }

    int batch_shape[] = {image_count, 1, 28, 28};
    Tensor* batch_tensor = tensor_create(batch_shape, 4);
    if (!batch_tensor) {
        fprintf(stderr, "Error: Failed to create batch tensor\n");
        free(image_list);
        tensor_free(grid_tensor);
        return NULL;
    }

    bool* valid_images = (bool*)malloc(sizeof(bool) * image_count);
    if (!valid_images) {
        fprintf(stderr, "Error: Failed to allocate valid_images array\n");
        free(image_list);
        tensor_free(batch_tensor);
        tensor_free(grid_tensor);
        return NULL;
    }
    memset(valid_images, 0, sizeof(bool) * image_count);

    int valid_count = 0;
    for (int i = 0; i < image_count; i++) {
        ImageInfo* info = &image_list[i];

        Image img;
        load_image(info->filepath, &img);

        if (!img.gray_pixels && !img.rgba_pixels) {
            fprintf(stderr, "Warning: Failed to load image: %s\n", info->filepath);
            continue;
        }

        if (!img.is_grayscale) {
            convert_to_grayscale(&img);
        }

        Image resized_img;
        if (img.width != 28 || img.height != 28) {
            resize_grayscale_image(&img, &resized_img, 28, 28);
            free_image(&img);
            img = resized_img;
        }

        Tensor* img_tensor = to_tensor(&img);
        if (!img_tensor) {
            fprintf(stderr, "Warning: Failed to convert image to tensor: %s\n", info->filepath);
            free_image(&img);
            continue;
        }

        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                float pixel_value = img_tensor->data[y * 28 + x];
                pixel_value = (pixel_value - 0.5f) / 0.5f;
                batch_tensor->data[valid_count * 28 * 28 + y * 28 + x] = pixel_value;
            }
        }

        valid_images[i] = true;
        valid_count++;

        tensor_free(img_tensor);
        free_image(&img);
    }

    if (valid_count == 0) {
        fprintf(stderr, "Warning: No valid images could be processed\n");
        free(valid_images);
        tensor_free(batch_tensor);
        Grid* grid = (Grid*)malloc(sizeof(Grid));
        if (!grid) {
            fprintf(stderr, "Error: Failed to allocate Grid structure\n");
            tensor_free(grid_tensor);
            return NULL;
        }
        grid->height = height;
        grid->width = width;
        grid->grid = grid_tensor;
        return grid;
    }

    if (valid_count < image_count) {
        int new_batch_shape[] = {valid_count, 1, 28, 28};
        Tensor* new_batch_tensor = tensor_create(new_batch_shape, 4);
        if (!new_batch_tensor) {
            fprintf(stderr, "Error: Failed to create resized batch tensor\n");
            free(valid_images);
            tensor_free(batch_tensor);
            tensor_free(grid_tensor);
            return NULL;
        }

        memcpy(new_batch_tensor->data, batch_tensor->data, sizeof(float) * valid_count * 28 * 28);
        tensor_free(batch_tensor);
        batch_tensor = new_batch_tensor;
    }

    CNNForwardResult* batch_forward_result = cnn_forward(model, batch_tensor);
    if (!batch_forward_result) {
        fprintf(stderr, "Error: Batched CNN forward pass failed\n");
        free(valid_images);
        tensor_free(batch_tensor);
        tensor_free(grid_tensor);
        return NULL;
    }

    Tensor* batch_softmax_output = softmax(batch_forward_result->fc2_out);
    if (!batch_softmax_output) {
        fprintf(stderr, "Error: Batch softmax failed\n");
        cnn_forward_result_free(batch_forward_result);
        free(valid_images);
        tensor_free(batch_tensor);
        tensor_free(grid_tensor);
        return NULL;
    }

    int batch_idx = 0;
    for (int i = 0; i < image_count; i++) {
        if (!valid_images[i]) continue;

        ImageInfo* info = &image_list[i];

        int base_idx = info->row * (width * 26) + info->col * 26;
        for (int c = 0; c < 26; c++) {
            grid_tensor->data[base_idx + c] = batch_softmax_output->data[batch_idx * 26 + c];
        }

        batch_idx++;
    }

    tensor_free(batch_softmax_output);
    cnn_forward_result_free(batch_forward_result);
    tensor_free(batch_tensor);
    free(valid_images);
    free(image_list);

    Grid* grid = (Grid*)malloc(sizeof(Grid));
    if (!grid) {
        fprintf(stderr, "Error: Failed to allocate Grid structure\n");
        tensor_free(grid_tensor);
        return NULL;
    }

    grid->height = height;
    grid->width = width;
    grid->grid = grid_tensor;

    return grid;
}

void FreeGrid(Grid* grid) {
    if (grid) {
        if (grid->grid) {
            tensor_free(grid->grid);
        }
        free(grid);
    }
}

WordsArray* create_words_array(const char* words_path, CNN* model) {
    if (!words_path || !model) {
        fprintf(stderr, "Error: Invalid parameters to create_words_array\n");
        return NULL;
    }

    DIR* dir = opendir(words_path);
    if (!dir) {
        fprintf(stderr, "Error: Failed to open words directory: %s\n", words_path);
        return NULL;
    }

    WordsArray* words_array = (WordsArray*)malloc(sizeof(WordsArray));
    if (!words_array) {
        fprintf(stderr, "Error: Failed to allocate WordsArray\n");
        closedir(dir);
        return NULL;
    }

    words_array->words = NULL;
    words_array->count = 0;
    words_array->capacity = 0;

    struct dirent* entry;
    char filepath[1024];

    int word_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "word_", 5) == 0 &&
            strstr(entry->d_name, ".png") &&
            !strstr(entry->d_name, "letter")) {
            word_count++;
        }
    }

    rewinddir(dir);

    if (word_count > 0) {
        words_array->words = (Word*)malloc(sizeof(Word) * word_count);
        if (!words_array->words) {
            fprintf(stderr, "Error: Failed to allocate words array\n");
            free(words_array);
            closedir(dir);
            return NULL;
        }
        words_array->capacity = word_count;
    }

    typedef struct {
        Tensor* letters;
        Tensor* probabilities;
        int letter_offset;
    } WordData;

    WordData* word_data_array = (WordData*)malloc(sizeof(WordData) * word_count);
    if (!word_data_array) {
        fprintf(stderr, "Error: Failed to allocate word data array\n");
        free(words_array->words);
        free(words_array);
        closedir(dir);
        return NULL;
    }

    int total_letters = 0;
    int word_idx = 0;

    char **image_names = NULL;
    if (word_count > 0) {
        image_names = (char**)malloc(sizeof(char*) * word_count);
        if (!image_names) {
            fprintf(stderr, "Error: Failed to allocate image_names array\n");
            for (int i = 0; i < word_idx; i++) {
                if (word_data_array[i].letters) {
                    tensor_free(word_data_array[i].letters);
                }
            }
            free(word_data_array);
            if (words_array->words) {
                free(words_array->words);
            }
            free(words_array);
            closedir(dir);
            return NULL;
        }
        memset(image_names, 0, sizeof(char*) * word_count);
    }

    rewinddir(dir);

    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "word_", 5) == 0 &&
            strstr(entry->d_name, ".png") &&
            !strstr(entry->d_name, "letter")) {

            snprintf(filepath, sizeof(filepath), "%s/%s", words_path, entry->d_name);

            Image word_image;
            load_image(filepath, &word_image);
            if (!word_image.gray_pixels && !word_image.rgba_pixels) {
                fprintf(stderr, "Warning: Failed to load word image: %s\n", filepath);
                continue;
            }

            if (!word_image.is_grayscale) {
                convert_to_grayscale(&word_image);
            }

            Tensor* word_letters = extract_word_letters(&word_image);
            if (!word_letters) {
                fprintf(stderr, "Warning: Failed to extract letters from: %s\n", filepath);
                free_image(&word_image);
                continue;
            }

            word_data_array[word_idx].letters = word_letters;
            word_data_array[word_idx].probabilities = NULL;
            word_data_array[word_idx].letter_offset = total_letters;
            image_names[word_idx] = malloc(strlen(entry->d_name) + 1);
            if (image_names[word_idx]) {
                strcpy(image_names[word_idx], entry->d_name);
            }

            total_letters += word_letters->shape[0];
            word_idx++;

            free_image(&word_image);
        }
    }

    if (total_letters > 0) {
        int batch_shape[] = {total_letters, 1, 28, 28};
        Tensor* batched_input = tensor_create_zero(batch_shape, 4);
        if (!batched_input) {
            fprintf(stderr, "Error: Failed to create batched CNN input tensor\n");
            for (int i = 0; i < word_idx; i++) {
                if (word_data_array[i].letters) {
                    tensor_free(word_data_array[i].letters);
                }
            }
            free(word_data_array);
            free(words_array->words);
            free(words_array);
            closedir(dir);
            return NULL;
        }

        int global_letter_idx = 0;
        for (int w = 0; w < word_idx; w++) {
            Tensor* word_letters = word_data_array[w].letters;
            int num_letters = word_letters->shape[0];

            for (int i = 0; i < num_letters; i++) {
                for (int y = 0; y < 28; y++) {
                    for (int x = 0; x < 28; x++) {
                        float pixel_value = word_letters->data[i * 28 * 28 + y * 28 + x];
                        pixel_value = (pixel_value - 0.5f) / 0.5f;
                        int batch_idx = global_letter_idx * 28 * 28 + y * 28 + x;
                        batched_input->data[batch_idx] = pixel_value;
                    }
                }
                global_letter_idx++;
            }
        }

        CNNForwardResult* batch_forward_result = cnn_forward(model, batched_input);
        if (!batch_forward_result) {
            fprintf(stderr, "Error: Batched CNN forward pass failed\n");
            tensor_free(batched_input);
            for (int i = 0; i < word_idx; i++) {
                if (word_data_array[i].letters) {
                    tensor_free(word_data_array[i].letters);
                }
            }
            free(word_data_array);
            free(words_array->words);
            free(words_array);
            closedir(dir);
            return NULL;
        }

        Tensor* batch_softmax_output = softmax(batch_forward_result->fc2_out);
        if (!batch_softmax_output) {
            fprintf(stderr, "Error: Batch softmax failed\n");
            cnn_forward_result_free(batch_forward_result);
            tensor_free(batched_input);
            for (int i = 0; i < word_idx; i++) {
                if (word_data_array[i].letters) {
                    tensor_free(word_data_array[i].letters);
                }
            }
            free(word_data_array);
            free(words_array->words);
            free(words_array);
            closedir(dir);
            return NULL;
        }

        for (int w = 0; w < word_idx; w++) {
            Tensor* word_letters = word_data_array[w].letters;
            int num_letters = word_letters->shape[0];
            int letter_offset = word_data_array[w].letter_offset;

            int prob_shape[] = {num_letters, 26};
            Tensor* word_probabilities = tensor_create_zero(prob_shape, 2);
            if (!word_probabilities) {
                fprintf(stderr, "Warning: Failed to allocate word probability tensor\n");
                continue;
            }

            for (int i = 0; i < num_letters; i++) {
                int global_letter_idx = letter_offset + i;
                int base_idx = i * 26;
                for (int c = 0; c < 26; c++) {
                    word_probabilities->data[base_idx + c] = batch_softmax_output->data[global_letter_idx * 26 + c];
                }
            }

            Word word_struct;
            word_struct.probabilities = word_probabilities;
            char* full_name = image_names[w];
            if (full_name) {
                size_t name_len = strlen(full_name);
                char* dot = strrchr(full_name, '.');
                if (dot) {
                    name_len = dot - full_name;
                }
                word_struct.image_name = (char*)malloc(name_len + 1);
                if (word_struct.image_name) {
                    strncpy(word_struct.image_name, full_name, name_len);
                    word_struct.image_name[name_len] = '\0';
                } else {
                    word_struct.image_name = NULL;
                    fprintf(stderr, "Warning: Failed to allocate memory for image name\n");
                }
            } else {
                word_struct.image_name = NULL;
            }

            words_array->words[words_array->count] = word_struct;
            words_array->count++;

            word_data_array[w].probabilities = word_probabilities;
        }

        tensor_free(batch_softmax_output);
        cnn_forward_result_free(batch_forward_result);
        tensor_free(batched_input);
    }

    for (int i = 0; i < word_idx; i++) {
        if (word_data_array[i].letters) {
            tensor_free(word_data_array[i].letters);
        }
    }
    free(word_data_array);

    if (image_names) {
        for (int i = 0; i < word_count; i++) {
            if (image_names[i]) {
                free(image_names[i]);
            }
        }
        free(image_names);
    }

    closedir(dir);
    return words_array;
}

void FreeWordsArray(WordsArray* words_array) {
    if (words_array) {
        if (words_array->words) {
            for (int i = 0; i < words_array->count; i++) {
                if (words_array->words[i].probabilities) {
                    tensor_free(words_array->words[i].probabilities);
                }
                if (words_array->words[i].image_name) {
                    free(words_array->words[i].image_name);
                }
            }
            free(words_array->words);
        }
        free(words_array);
    }
}

char* word_to_string(const Word* word) {
    if (!word || !word->probabilities) {
        return NULL;
    }

    int num_letters = word->probabilities->shape[0];
    char* word_str = (char*)malloc(sizeof(char) * (num_letters + 1));
    if (!word_str) {
        return NULL;
    }

    for (int i = 0; i < num_letters; i++) {
        int base_idx = i * 26;
        int max_class = 0;
        float max_prob = word->probabilities->data[base_idx];

        for (int c = 1; c < 26; c++) {
            float prob = word->probabilities->data[base_idx + c];
            if (prob > max_prob) {
                max_prob = prob;
                max_class = c;
            }
        }

        word_str[i] = 'A' + max_class;
    }

    word_str[num_letters] = '\0';
    return word_str;
}

char* grid_to_string(const Grid* grid) {
    if (!grid || !grid->grid) {
        return NULL;
    }

    int height = grid->height;
    int width = grid->width;

    char* grid_str = (char*)malloc(sizeof(char) * (height * (width + 1) + 1));
    if (!grid_str) {
        return NULL;
    }

    int str_idx = 0;
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            int base_idx = r * (width * 26) + c * 26;
            int max_class = 0;
            float max_prob = grid->grid->data[base_idx];

            for (int letter = 1; letter < 26; letter++) {
                float prob = grid->grid->data[base_idx + letter];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_class = letter;
                }
            }

            grid_str[str_idx++] = 'A' + max_class;
        }
        grid_str[str_idx++] = '\n';
    }

    grid_str[str_idx] = '\0';
    return grid_str;
}

int char_to_index(char c) {
    return (int)(c - 'A');
}

char index_to_char(int idx) {
    return (char)(idx + 'A');
}

Tensor* create_log_prob_tensor(Tensor* prob_tensor) {
    if (!prob_tensor) {
        return NULL;
    }

    Tensor* log_tensor = tensor_create(prob_tensor->shape, prob_tensor->ndim);
    if (!log_tensor) {
        return NULL;
    }

    for (int i = 0; i < prob_tensor->size; i++) {
        float prob = prob_tensor->data[i];
        if (prob < 1e-9f) {
            prob = 1e-9f;
        }
        log_tensor->data[i] = logf(prob);
    }

    return log_tensor;
}

WordMatch* find_best_word_match(const Grid* grid, const Word* word, const char* word_str) {
    if (!grid || !word || !word_str || !grid->grid || !word->probabilities) {
        return NULL;
    }

    int height = grid->height;
    int width = grid->width;
    int word_len = word->probabilities->shape[0];

    Tensor* log_grid = create_log_prob_tensor(grid->grid);

    if (!log_grid) {
        tensor_free(log_grid);
        return NULL;
    }

    Tensor* letter_scores = tensor_create((int[]){height, width, word_len}, 3);
    if (!letter_scores) {
        tensor_free(log_grid);
        return NULL;
    }

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            for (int i = 0; i < word_len; i++) {
                float score = 0.0f;
                int grid_base = r * (width * 26) + c * 26;
                int word_base = i * 26;

                for (int k = 0; k < 26; k++) {
                    score += word->probabilities->data[word_base + k] * log_grid->data[grid_base + k];
                }

                int score_idx = r * (width * word_len) + c * word_len + i;
                letter_scores->data[score_idx] = score;
            }
        }
    }

    typedef struct {
        int dr, dc;
        const char* name;
    } Direction;

    Direction directions[] = {
        {0, 1, "Right"},
        {0, -1, "Left"},
        {1, 0, "Down"},
        {-1, 0, "Up"},
        {1, 1, "DownRight"},
        {1, -1, "DownLeft"},
        {-1, 1, "UpRight"},
        {-1, -1, "UpLeft"}
    };
    int num_directions = sizeof(directions) / sizeof(directions[0]);

    float best_score = -INFINITY;
    WordMatch* best_match = NULL;

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            for (int d = 0; d < num_directions; d++) {
                Direction dir = directions[d];

                float current_score = 0.0f;
                Position* path_positions = (Position*)malloc(sizeof(Position) * word_len);
                if (!path_positions) {
                    continue;
                }

                bool valid_path = true;

                for (int k = 0; k < word_len; k++) {
                    int rr = r + k * dir.dr;
                    int cc = c + k * dir.dc;

                    if (rr < 0 || rr >= height || cc < 0 || cc >= width) {
                        valid_path = false;
                        break;
                    }

                    int score_idx = rr * (width * word_len) + cc * word_len + k;
                    current_score += letter_scores->data[score_idx];

                    path_positions[k].row = rr;
                    path_positions[k].col = cc;
                }

                if (valid_path && current_score > best_score) {
                    best_score = current_score;

                    if (best_match) {
                        free_word_match(best_match);
                    }

                    best_match = (WordMatch*)malloc(sizeof(WordMatch));
                    if (!best_match) {
                        free(path_positions);
                        continue;
                    }

                    best_match->word_str = (char*)malloc(strlen(word_str) + 1);
                    if (best_match->word_str) {
                        strcpy(best_match->word_str, word_str);
                    }
                    best_match->start_pos.row = r;
                    best_match->start_pos.col = c;
                    strcpy(best_match->direction, dir.name);
                    best_match->log_prob_score = current_score;

                    best_match->path.positions = path_positions;
                    best_match->path.length = word_len;
                } else {
                    free(path_positions);
                }
            }
        }
    }

    tensor_free(log_grid);
    tensor_free(letter_scores);

    return best_match;
}

void free_word_match(WordMatch* match) {
    if (match) {
        if (match->word_str) {
            free(match->word_str);
        }
        if (match->path.positions) {
            free(match->path.positions);
        }
        free(match);
    }
}
