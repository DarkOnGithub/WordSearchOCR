#pragma once

#include "../image/operations.h"
#include <stdint.h>
#include <stdbool.h>

/*
    Data structure for a bounding box (x, y, width, height)
*/
typedef struct {
    int x, y, width, height;
} BoundingBox;

/*
    Dynamic array of bounding boxes
*/
typedef struct {
    BoundingBox* boxes;
    int count;
    int capacity;
} BoundingBoxArray;

/*
    Dynamic array of bounding box groups
*/
typedef struct {
    BoundingBoxArray** groups;
    int count;
    int capacity;
} WordGroups;

/*
    Initialize a bounding box array
*/
void initBoundingBoxArray(BoundingBoxArray* array);

/*
    Add a bounding box to the array
*/
bool addBoundingBox(BoundingBoxArray* array, BoundingBox box);

/*
    Free a bounding box array
*/
void freeBoundingBoxArray(BoundingBoxArray* array);

/*
    Initialize word groups
*/
void initWordGroups(WordGroups* groups);

/*
    Add a group to word groups
*/
bool addWordGroup(WordGroups* groups, BoundingBoxArray* group);

/*
    Free word groups
*/
void freeWordGroups(WordGroups* groups);

/*
    Merge nearby bounding boxes based on gap and vertical overlap criteria
*/
BoundingBoxArray* merge_nearby_boxes(const BoundingBoxArray* boxes, int max_gap, float max_vertical_overlap);

/*
    Find word groups from bounding boxes by analyzing row alignment
*/
WordGroups* find_word_groups(const BoundingBoxArray* boxes, int max_distance, int alignment_threshold);

/*
    Select the main word group (largest group, or all words if no large group found)
*/
BoundingBoxArray* select_main_word_group(const WordGroups* groups);

/*
    Main word detection function that processes an image and returns detected word bounding boxes
*/
BoundingBoxArray* detect_words(const char* image_path, const char* debug_prefix);

/*
    Draw bounding boxes on an image
*/
void draw_bounding_boxes(Image* image, const BoundingBoxArray* boxes, uint32_t color, int thickness);
