#pragma once

#include "../image/image.h"
#include <stdint.h>


/*
    Apply a Gaussian blur to an image.
    !Warning: Works on both grayscale. Modifies the image in place.
    Ressource: https://en.wikipedia.org/wiki/Gaussian_blur
*/
void gaussian_blur(Image* image, uint8_t kernel_size, float sigma);

/*
    Apply adaptive thresholding to a grayscale image.
    Resource: https://docs.opencv.org/4.x/d7/dd0/tutorial_js_thresholding.html
*/
void adaptiveThreshold(Image* image, uint8_t max_value, int method, int threshold_type,
                      int block_size, double c);

/*
    Apply Otsu's thresholding to a grayscale image.
    Resource: https://en.wikipedia.org/wiki/Otsu%27s_method
*/
double threshold(Image* image, uint8_t max_value);

/*
    Correct binary image orientation by ensuring foreground is white (255) on black (0) background.
    !Warning: Only works on grayscale images. Modifies the image in place.
*/
int correctBinaryImageOrientation(Image* image);

typedef struct {
    int rows;
    int cols;
    uint8_t* data;
} StructuringElement;

// Structuring element shapes (compatible with OpenCV)
#define MORPH_RECT   0  // Rectangular structuring element
#define MORPH_CROSS  1  // Cross-shaped structuring element
#define MORPH_ELLIPSE 2  // Elliptical structuring element

StructuringElement* getStructuringElement(int shape, int ksize_width, int ksize_height);

typedef enum MorphologicalOperation {
    MORPH_OPEN,
    MORPH_CLOSE,
    MORPH_ERODE,
    MORPH_DILATE
} MorphologicalOperation;

void morphologyEx(Image* image, MorphologicalOperation operation, StructuringElement* kernel, int iterations);
void freeStructuringElement(StructuringElement* kernel);
void add(Image* src1, Image* src2, Image* dst);
void bitwise_or(const Image* src1, const Image* src2, Image* dst);

typedef struct {
    int x, y;
} Point;

typedef struct Contour {
    Point* points;
    int count;
    int capacity;
} Contour;

typedef struct {
    Contour* contours;
    int count;
    int capacity;
} Contours;


/*
    Find contours in a binary image.
    !Warning: Only works on grayscale images. Caller must free the returned structure.
    Resource: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
*/
Contours* findContours(const Image* image, int mode);
void freeContours(Contours* contours);

/*
    Calculate the arc length (perimeter) of a contour.
    Resource: https://en.wikipedia.org/wiki/Contour_(computer_vision)
*/
double arcLength(const Contour* contour, int closed);

/*
    Filter contours based on minimum arc length.
*/
Contours* filterContoursByLength(const Contours* contours, double min_length);

typedef struct {
    int x, y;
    int width;
    int height;
} Rect;

/*
    Calculate the bounding rectangle of a contour.
*/
int boundingRect(const Contour* contour, Rect* rect);

/*
    Find the contour with maximum area (approximated by bounding rectangle area).
*/
int findMaxAreaContour(const Contours* contours);

/*
    Find the bounding rectangle that contains all given rectangles with padding.
*/
int getBoundingRectOfRects(const Rect* rects, int count, int padding, int image_width, int image_height, Rect* result);