#include "image/operations.h"
#include "image/image.h"
#include <cairo/cairo.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M_PI 3.14159265358979323846

/*
    Create a 2D Gaussian kernel for convolution.
*/
static float *create_gaussian_kernel(int kernel_size, float sigma)
{
    if (kernel_size % 2 == 0 || kernel_size < 1)
    {
        return NULL;
    }

    float *kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    if (!kernel)
    {
        return NULL;
    }

    int radius = kernel_size / 2;
    float sum = 0.0f;
    float two_sigma_sq = 2.0f * sigma * sigma;

    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            float exponent = -(x * x + y * y) / two_sigma_sq;
            float value = expf(exponent);
            kernel[(y + radius) * kernel_size + (x + radius)] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernel_size * kernel_size; i++)
    {
        kernel[i] /= sum;
    }

    return kernel;
}

/*
    Apply a Gaussian blur to an image.
    !Warning: Works on both grayscale. Modifies the image in place.
    Ressource: https://en.wikipedia.org/wiki/Gaussian_blur
*/
void gaussian_blur(Image *image, uint8_t kernel_size, float sigma)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for gaussian_blur\n");
        return;
    }

    if ((image->is_grayscale && !image->gray_pixels) ||
        (!image->is_grayscale && !image->rgba_pixels))
    {
        fprintf(stderr, "Error: Image has no pixel data\n");
        return;
    }

    if (kernel_size % 2 == 0)
    {
        fprintf(stderr, "Error: Kernel size must be odd\n");
        return;
    }

    // If sigma is 0, calculate it automatically like OpenCV does
    float actual_sigma = sigma;
    if (actual_sigma == 0.0f)
    {
        actual_sigma = 0.3f * ((kernel_size - 1) * 0.5f - 1.0f) + 0.8f;
        if (actual_sigma < 0.0f)
            actual_sigma = 0.8f; // Ensure minimum sigma
    }

    float *kernel = create_gaussian_kernel(kernel_size, actual_sigma);
    if (!kernel)
    {
        fprintf(stderr, "Error: Failed to create Gaussian kernel\n");
        return;
    }

    int radius = kernel_size / 2;

    if (image->is_grayscale)
    {
        uint8_t *new_data =
            malloc(image->width * image->height * sizeof(uint8_t));
        if (!new_data)
        {
            fprintf(stderr, "Error: Failed to allocate new data buffer\n");
            free(kernel);
            return;
        }

        for (int y = 0; y < image->height; y++)
        {
            for (int x = 0; x < image->width; x++)
            {
                float sum = 0.0f;
                for (int ky = -radius; ky <= radius; ky++)
                {
                    for (int kx = -radius; kx <= radius; kx++)
                    {
                        int img_x = x + kx;
                        int img_y = y + ky;
                        if (img_x < 0)
                            img_x = 0;
                        if (img_x >= image->width)
                            img_x = image->width - 1;
                        if (img_y < 0)
                            img_y = 0;
                        if (img_y >= image->height)
                            img_y = image->height - 1;
                        int img_idx = img_y * image->width + img_x;
                        int kernel_idx =
                            (ky + radius) * kernel_size + (kx + radius);
                        sum += image->gray_pixels[img_idx] * kernel[kernel_idx];
                    }
                }
                int idx = y * image->width + x;
                new_data[idx] = (uint8_t)sum;
            }
        }
        free(image->gray_pixels);
        image->gray_pixels = new_data;
    }
    free(kernel);

    printf("Applied Gaussian blur (kernel_size=%d, sigma=%.2f)\n", kernel_size,
           actual_sigma);
}

/*
    Apply adaptive thresholding to a grayscale image.
    Resource: https://docs.opencv.org/4.x/d7/dd0/tutorial_js_thresholding.html
*/
void adaptiveThreshold(Image *image, uint8_t max_value, int method,
                       int threshold_type, int block_size, double c)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for adaptiveThreshold\n");
        return;
    }

    if (!image->is_grayscale || !image->gray_pixels)
    {
        fprintf(stderr,
                "Error: Adaptive thresholding requires a grayscale image\n");
        return;
    }

    if (block_size < 3 || block_size % 2 == 0)
    {
        fprintf(stderr, "Error: block_size must be odd and >= 3\n");
        return;
    }

    if (method != 0 && method != 1)
    {
        fprintf(stderr, "Error: method must be 0 (mean) or 1 (gaussian)\n");
        return;
    }

    if (threshold_type != 0 && threshold_type != 1)
    {
        fprintf(stderr,
                "Error: threshold_type must be 0 (binary) or 1 (binary_inv)\n");
        return;
    }

    int width = image->width;
    int height = image->height;
    int radius = block_size / 2;
    uint8_t *new_pixels = malloc(width * height * sizeof(uint8_t));

    if (!new_pixels)
    {
        fprintf(
            stderr,
            "Error: Failed to allocate memory for adaptive threshold output\n");
        return;
    }

    float *gaussian_kernel = NULL;
    // Gaussian method
    if (method == 1)
    {
        gaussian_kernel =
            create_gaussian_kernel(block_size, (float)block_size / 6.0f);
        if (!gaussian_kernel)
        {
            fprintf(stderr, "Error: Failed to create gaussian kernel for "
                            "adaptive threshold\n");
            free(new_pixels);
            return;
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            double sum = 0.0;
            double weight_sum = 0.0;

            for (int ky = -radius; ky <= radius; ky++)
            {
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (nx < 0)
                        nx = 0;
                    if (nx >= width)
                        nx = width - 1;
                    if (ny < 0)
                        ny = 0;
                    if (ny >= height)
                        ny = height - 1;

                    int nidx = ny * width + nx;
                    uint8_t pixel_value = image->gray_pixels[nidx];
                    // Mean method
                    if (method == 0)
                    {
                        sum += pixel_value;
                        weight_sum += 1.0;
                    }
                    else
                    { // Gaussian method
                        int kernel_idx =
                            (ky + radius) * block_size + (kx + radius);
                        float weight = gaussian_kernel[kernel_idx];
                        sum += pixel_value * weight;
                        weight_sum += weight;
                    }
                }
            }

            double threshold = (sum / weight_sum) - c;
            if (threshold < 0)
                threshold = 0;
            if (threshold > 255)
                threshold = 255;

            uint8_t pixel_value = image->gray_pixels[idx];
            uint8_t result;

            if (threshold_type == 0)
            { // THRESH_BINARY
                result = (pixel_value > threshold) ? max_value : 0;
            }
            else
            { // THRESH_BINARY_INV
                result = (pixel_value > threshold) ? 0 : max_value;
            }

            new_pixels[idx] = result;
        }
    }

    free(image->gray_pixels);
    image->gray_pixels = new_pixels;

    if (gaussian_kernel)
    {
        free(gaussian_kernel);
    }

    printf("Applied adaptive threshold (method=%s, threshold_type=%s, "
           "block_size=%d, C=%.1f)\n",
           method == 0 ? "mean" : "gaussian",
           threshold_type == 0 ? "binary" : "binary_inv", block_size, c);
}

/*
    Apply Otsu's thresholding to a grayscale image.
    Resource: https://en.wikipedia.org/wiki/Otsu%27s_method
*/
double threshold(Image *image, uint8_t max_value)
{
    if (!image || !image->gray_pixels)
    {
        fprintf(stderr,
                "Error: Otsu thresholding requires a grayscale image\n");
        return -1.0;
    }

    int width = image->width;
    int height = image->height;
    uint8_t *pixels = image->gray_pixels;
    int total_pixels = width * height;

    int histogram[256] = {0};
    for (int i = 0; i < total_pixels; i++)
    {
        histogram[pixels[i]]++;
    }

    double sum = 0.0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * histogram[i];
    }

    double sum_b = 0.0; // Sum of background pixels
    int w_b = 0;        // Weight of background pixels
    int w_f = 0;        // Weight of foreground pixels

    double max_variance = 0.0;
    double best_threshold = 0.0;

    for (int t = 0; t < 256; t++)
    {
        w_b += histogram[t];
        if (w_b == 0)
            continue;

        w_f = total_pixels - w_b;
        if (w_f == 0)
            break;

        sum_b += (double)t * histogram[t];

        double m_b = sum_b / w_b;
        double m_f = (sum - sum_b) / w_f;

        // Between-class variance
        double variance = (double)w_b * (double)w_f * (m_b - m_f) * (m_b - m_f);

        if (variance > max_variance)
        {
            max_variance = variance;
            best_threshold = t;
        }
    }

    for (int i = 0; i < total_pixels; i++)
    {
        pixels[i] = (pixels[i] > best_threshold) ? max_value : 0;
    }

    printf("Applied Otsu's threshold with value %.2f\n", best_threshold);
    return best_threshold;
}

/*
    Correct binary image orientation by ensuring foreground is white (255) on
   black (0) background. !Warning: Only works on grayscale images. Modifies the
   image in place.
*/
int correctBinaryImageOrientation(Image *image)
{
    if (!image || !image->gray_pixels)
    {
        fprintf(stderr, "Error: Binary image orientation correction requires a "
                        "grayscale image\n");
        return -1;
    }

    int total_pixels = image->width * image->height;
    double mean_value = 0.0;

    for (int i = 0; i < total_pixels; i++)
    {
        mean_value += image->gray_pixels[i];
    }
    mean_value /= total_pixels;

    if (mean_value > 127.0)
    {
        for (int i = 0; i < total_pixels; i++)
        {
            image->gray_pixels[i] = 255 - image->gray_pixels[i];
        }
        return 1;
    }
    else
    {
        return 0;
    }
}

/*
    Create a structuring element of the specified shape and size.
    A structuring element (also called a kernel) is a small matrix used in
   morphological operations on images. Resource:
   https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
*/
StructuringElement *getStructuringElement(int shape, int ksize_width,
                                          int ksize_height)
{
    if (ksize_width <= 0 || ksize_height <= 0)
    {
        fprintf(stderr, "Error: Invalid structuring element size\n");
        return NULL;
    }

    if (shape < 0 || shape > 2)
    {
        fprintf(stderr, "Error: Invalid shape type (must be 0-2)\n");
        return NULL;
    }

    StructuringElement *kernel = malloc(sizeof(StructuringElement));
    if (!kernel)
    {
        fprintf(stderr, "Error: Failed to allocate structuring element\n");
        return NULL;
    }

    kernel->rows = ksize_height;
    kernel->cols = ksize_width;
    kernel->data = calloc(ksize_width * ksize_height, sizeof(uint8_t));

    if (!kernel->data)
    {
        fprintf(stderr, "Error: Failed to allocate structuring element data\n");
        free(kernel);
        return NULL;
    }

    if (shape == 0)
    { // MORPH_RECT -> all pixels are 1
        memset(kernel->data, 1, ksize_width * ksize_height * sizeof(uint8_t));
    }
    else if (shape == 1)
    { // MORPH_CROSS -> plus sign shape (full center row and column)
        int center_x = ksize_width / 2;
        int center_y = ksize_height / 2;

        for (int x = 0; x < ksize_width; x++)
        {
            kernel->data[center_y * ksize_width + x] = 1;
        }
        for (int y = 0; y < ksize_height; y++)
        {
            kernel->data[y * ksize_width + center_x] = 1;
        }
    }
    else if (shape == 2)
    { // MORPH_ELLIPSE -> diamond shape (Chebyshev distance)
        int center_x = ksize_width / 2;
        int center_y = ksize_height / 2;
        int radius =
            (ksize_width < ksize_height ? ksize_width : ksize_height) / 2;

        for (int y = 0; y < ksize_height; y++)
        {
            for (int x = 0; x < ksize_width; x++)
            {
                // https://en.wikipedia.org/wiki/Chebyshev_distance
                int dx = abs(x - center_x);
                int dy = abs(y - center_y);
                if (dx <= radius && dy <= radius)
                {
                    kernel->data[y * ksize_width + x] = 1;
                }
            }
        }
    }

    return kernel;
}

void freeStructuringElement(StructuringElement *kernel)
{
    if (kernel)
    {
        if (kernel->data)
        {
            free(kernel->data);
        }
        free(kernel);
    }
}

/*
    Apply erosion operation to a grayscale image.
    Erosion is a morphological operation that removes small objects from the
   foreground (usually white pixels) by shrinking them. Resource:
   https://en.wikipedia.org/wiki/Erosion_(morphology)
*/
static void erode(const uint8_t *src, uint8_t *dst, int width, int height,
                  StructuringElement *kernel)
{
    int k_center_x = kernel->cols / 2;
    int k_center_y = kernel->rows / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t min_val = 255;

            for (int ky = 0; ky < kernel->rows; ky++)
            {
                for (int kx = 0; kx < kernel->cols; kx++)
                {
                    if (kernel->data[ky * kernel->cols + kx])
                    {
                        int img_x = x + (kx - k_center_x);
                        int img_y = y + (ky - k_center_y);

                        if (img_x < 0)
                            img_x = 0;
                        if (img_x >= width)
                            img_x = width - 1;
                        if (img_y < 0)
                            img_y = 0;
                        if (img_y >= height)
                            img_y = height - 1;

                        uint8_t pixel = src[img_y * width + img_x];
                        if (pixel < min_val)
                        {
                            min_val = pixel;
                        }
                    }
                }
            }
            dst[y * width + x] = min_val;
        }
    }
}

/*
    Apply dilation operation to a grayscale image.
    Dilation is a morphological operation that increases the size of objects in
   the foreground (usually white pixels) by expanding them. Resource:
   https://en.wikipedia.org/wiki/Dilation_(morphology)
*/
static void dilate(const uint8_t *src, uint8_t *dst, int width, int height,
                   StructuringElement *kernel)
{
    int k_center_x = kernel->cols / 2;
    int k_center_y = kernel->rows / 2;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            uint8_t max_val = 0;

            for (int ky = 0; ky < kernel->rows; ky++)
            {
                for (int kx = 0; kx < kernel->cols; kx++)
                {
                    if (kernel->data[ky * kernel->cols + kx])
                    {
                        int img_x = x + (kx - k_center_x);
                        int img_y = y + (ky - k_center_y);

                        if (img_x < 0)
                            img_x = 0;
                        if (img_x >= width)
                            img_x = width - 1;
                        if (img_y < 0)
                            img_y = 0;
                        if (img_y >= height)
                            img_y = height - 1;

                        uint8_t pixel = src[img_y * width + img_x];
                        if (pixel > max_val)
                        {
                            max_val = pixel;
                        }
                    }
                }
            }
            dst[y * width + x] = max_val;
        }
    }
}

/*
    Apply morphological operations to a grayscale image.
    Morphological operations are a set of image processing techniques that use
   mathematical morphology, which is a mathematical framework for analyzing and
   modifying geometric structures in images. !Warning: Only works on grayscale
   images. Modifies the image in place. Resource:
   https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
*/
void morphologyEx(Image *image, MorphologicalOperation operation,
                  StructuringElement *kernel, int iterations)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for morphologyEx\n");
        return;
    }

    if (!image->is_grayscale || !image->gray_pixels)
    {
        fprintf(stderr, "Error: MorphologyEx requires a grayscale image\n");
        return;
    }

    if (!kernel || !kernel->data)
    {
        fprintf(stderr, "Error: Invalid structuring element\n");
        return;
    }

    if (operation < MORPH_OPEN || operation > MORPH_DILATE)
    {
        fprintf(stderr, "Error: Invalid operation type (must be 0-3)\n");
        return;
    }

    if (iterations <= 0)
    {
        fprintf(stderr, "Error: Iterations must be positive\n");
        return;
    }

    int width = image->width;
    int height = image->height;
    int total_pixels = width * height;

    uint8_t *temp1 = malloc(total_pixels * sizeof(uint8_t));
    uint8_t *temp2 = malloc(total_pixels * sizeof(uint8_t));

    if (!temp1 || !temp2)
    {
        fprintf(
            stderr,
            "Error: Failed to allocate memory for morphological operation\n");
        if (temp1)
            free(temp1);
        if (temp2)
            free(temp2);
        return;
    }

    memcpy(temp1, image->gray_pixels, total_pixels * sizeof(uint8_t));

    const char *operation_names[] = {"open", "close", "erode", "dilate"};

    for (int iter = 0; iter < iterations; iter++)
    {
        switch (operation)
        {
        case MORPH_OPEN: // MORPH_OPEN: erode then dilate
            erode(temp1, temp2, width, height, kernel);
            dilate(temp2, temp1, width, height, kernel);
            break;

        case MORPH_CLOSE: // MORPH_CLOSE: dilate then erode
            dilate(temp1, temp2, width, height, kernel);
            erode(temp2, temp1, width, height, kernel);
            break;

        case MORPH_ERODE: // MORPH_ERODE: erode only
            erode(temp1, temp2, width, height, kernel);
            memcpy(temp1, temp2, total_pixels * sizeof(uint8_t));
            break;

        case MORPH_DILATE: // MORPH_DILATE: dilate only
            dilate(temp1, temp2, width, height, kernel);
            memcpy(temp1, temp2, total_pixels * sizeof(uint8_t));
            break;
        }
    }

    // Copy final result back to image
    memcpy(image->gray_pixels, temp1, total_pixels * sizeof(uint8_t));

    free(temp1);
    free(temp2);

    printf(
        "Applied morphological %s operation (kernel: %dx%d, iterations: %d)\n",
        operation_names[operation], kernel->cols, kernel->rows, iterations);
}

/*
    Add two grayscale images pixel-wise with saturation.
    !Warning: All images must be grayscale and have the same dimensions.
*/
void add(Image *src1, Image *src2, Image *dst)
{
    if (!src1 || !src2 || !dst)
    {
        fprintf(stderr, "Error: Invalid image(s) for add operation\n");
        return;
    }

    if (!src1->is_grayscale || !src1->gray_pixels)
    {
        fprintf(stderr, "Error: src1 must be a grayscale image\n");
        return;
    }

    if (!src2->is_grayscale || !src2->gray_pixels)
    {
        fprintf(stderr, "Error: src2 must be a grayscale image\n");
        return;
    }

    if (!dst->is_grayscale || !dst->gray_pixels)
    {
        fprintf(stderr, "Error: dst must be a grayscale image\n");
        return;
    }

    if (src1->width != src2->width || src1->height != src2->height ||
        src1->width != dst->width || src1->height != dst->height)
    {
        fprintf(stderr, "Error: All images must have the same dimensions\n");
        return;
    }

    int width = src1->width;
    int height = src1->height;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        int sum = (int)src1->gray_pixels[i] + (int)src2->gray_pixels[i];
        dst->gray_pixels[i] = (uint8_t)(sum > 255 ? 255 : sum);
    }

    printf("Applied pixel-wise addition with saturation (%dx%d)\n", width,
           height);
}

/*
    Bitwise OR operation for combining images pixel-wise.
    !Warning: All images must be grayscale and have the same dimensions.
*/
void bitwise_or(const Image *src1, const Image *src2, Image *dst)
{
    if (!src1 || !src2 || !dst)
    {
        fprintf(stderr, "Error: Invalid image(s) for bitwise_or operation\n");
        return;
    }

    if (!src1->is_grayscale || !src2->is_grayscale || !dst->is_grayscale)
    {
        fprintf(stderr, "Error: All images must be grayscale for bitwise_or\n");
        return;
    }

    if (!src1->gray_pixels || !src2->gray_pixels || !dst->gray_pixels)
    {
        fprintf(stderr, "Error: Images must have pixel data for bitwise_or\n");
        return;
    }

    if (src1->width != src2->width || src1->height != src2->height ||
        src1->width != dst->width || src1->height != dst->height)
    {
        fprintf(
            stderr,
            "Error: All images must have the same dimensions for bitwise_or\n");
        return;
    }

    int width = src1->width;
    int height = src1->height;
    int total_pixels = width * height;

    for (int i = 0; i < total_pixels; i++)
    {
        dst->gray_pixels[i] = src1->gray_pixels[i] | src2->gray_pixels[i];
    }

    printf("Applied bitwise OR operation (%dx%d)\n", width, height);
}

/*
    Add a point to a contour, expanding capacity if needed.
    Resource: https://en.wikipedia.org/wiki/Contour_(computer_vision)
*/
static int addPointToContour(Contour *contour, int x, int y)
{
    if (contour->count >= contour->capacity)
    {
        int new_capacity = contour->capacity == 0 ? 16 : contour->capacity * 2;
        Point *new_points =
            realloc(contour->points, new_capacity * sizeof(Point));
        if (!new_points)
        {
            return 0;
        }
        contour->points = new_points;
        contour->capacity = new_capacity;
    }

    contour->points[contour->count].x = x;
    contour->points[contour->count].y = y;
    contour->count++;
    return 1;
}

/*
    Add a contour to contours array, expanding capacity if needed.
*/
static int addContourToContours(Contours *contours, Contour contour)
{
    if (contours->count >= contours->capacity)
    {
        int new_capacity = contours->capacity == 0 ? 4 : contours->capacity * 2;
        Contour *new_contours =
            realloc(contours->contours, new_capacity * sizeof(Contour));
        if (!new_contours)
        {
            return 0;
        }
        contours->contours = new_contours;
        contours->capacity = new_capacity;
    }

    contours->contours[contours->count] = contour;
    contours->count++;
    return 1;
}

/*
    Check if a pixel is foreground (non-zero).
*/
static int isForeground(const Image *image, int x, int y)
{
    if (x < 0 || x >= image->width || y < 0 || y >= image->height)
    {
        return 0;
    }
    return image->gray_pixels[y * image->width + x] > 0;
}

/*
    Flood fill to find all connected foreground pixels in a component.
    Resource: https://en.wikipedia.org/wiki/Flood_fill
*/
static int floodFillComponent(const Image *image, int start_x, int start_y,
                              uint8_t *visited, Point *component_pixels,
                              int max_points)
{
    int width = image->width;
    int height = image->height;
    int count = 0;

    Point *stack = malloc(width * height * sizeof(Point));
    if (!stack)
    {
        return -1;
    }

    int stack_size = 0;

    stack[stack_size].x = start_x;
    stack[stack_size].y = start_y;
    stack_size++;

    visited[start_y * width + start_x] = 1;

    while (stack_size > 0 && count < max_points)
    {
        Point current = stack[--stack_size];

        if (count < max_points)
        {
            component_pixels[count++] = current;
        }

        int dx[4] = {0, 1, 0, -1};
        int dy[4] = {-1, 0, 1, 0};

        for (int i = 0; i < 4; i++)
        {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                int nidx = ny * width + nx;
                if (!visited[nidx] && isForeground(image, nx, ny))
                {
                    visited[nidx] = 1;
                    stack[stack_size].x = nx;
                    stack[stack_size].y = ny;
                    stack_size++;
                }
            }
        }
    }

    free(stack);
    return count;
}

/*
    Create a simple line representation from a set of connected pixels.
    For horizontal lines, find leftmost and rightmost points.
    For vertical lines, find topmost and bottommost points.
    Resource: https://en.wikipedia.org/wiki/Contour_(computer_vision)
*/
static int createLineContour(Point *component_pixels, int count,
                             Contour *contour)
{
    if (count < 2)
    {
        return 0;
    }

    contour->points = NULL;
    contour->count = 0;
    contour->capacity = 0;

    int min_x = component_pixels[0].x, max_x = component_pixels[0].x;
    int min_y = component_pixels[0].y, max_y = component_pixels[0].y;

    for (int i = 1; i < count; i++)
    {
        Point p = component_pixels[i];
        if (p.x < min_x)
            min_x = p.x;
        if (p.x > max_x)
            max_x = p.x;
        if (p.y < min_y)
            min_y = p.y;
        if (p.y > max_y)
            max_y = p.y;
    }

    int width = max_x - min_x + 1;
    int height = max_y - min_y + 1;

    if (width > height)
    {
        // Horizontal line: use leftmost and rightmost points
        addPointToContour(contour, min_x, min_y);
        addPointToContour(contour, max_x, max_y);
    }
    else
    {
        // Vertical line: use topmost and bottommost points
        addPointToContour(contour, min_x, min_y);
        addPointToContour(contour, max_x, max_y);
    }

    return 1;
}

/*
    Find contours in a binary image.
    !Warning: Only works on grayscale images. Caller must free the returned
   structure. Resource:
   https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
*/
Contours *findContours(const Image *image, int mode)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for findContours\n");
        return NULL;
    }

    if (!image->is_grayscale || !image->gray_pixels)
    {
        fprintf(stderr, "Error: findContours requires a grayscale image\n");
        return NULL;
    }

    if (mode != 0 && mode != 1)
    {
        fprintf(stderr, "Error: Only RETR_EXTERNAL (mode=0) and RETR_LIST "
                        "(mode=1) are supported\n");
        return NULL;
    }

    int width = image->width;
    int height = image->height;

    uint8_t *visited = calloc(width * height, sizeof(uint8_t));
    if (!visited)
    {
        fprintf(stderr, "Error: Failed to allocate visited array\n");
        return NULL;
    }

    Contours *contours = malloc(sizeof(Contours));
    if (!contours)
    {
        free(visited);
        fprintf(stderr, "Error: Failed to allocate contours structure\n");
        return NULL;
    }

    contours->contours = NULL;
    contours->count = 0;
    contours->capacity = 0;

    int max_component_pixels = width * height / 4;
    Point *component_pixels = malloc(max_component_pixels * sizeof(Point));
    if (!component_pixels)
    {
        free(visited);
        free(contours);
        return NULL;
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;

            if (visited[idx] || !isForeground(image, x, y))
            {
                continue;
            }

            // Find connected component using flood fill
            int pixel_count = floodFillComponent(
                image, x, y, visited, component_pixels, max_component_pixels);
            if (pixel_count < 0)
            {
                free(component_pixels);
                freeContours(contours);
                free(visited);
                return NULL;
            }

            Contour contour;
            if (pixel_count >= 2 &&
                createLineContour(component_pixels, pixel_count, &contour))
            {
                if (!addContourToContours(contours, contour))
                {
                    free(contour.points);
                    free(component_pixels);
                    freeContours(contours);
                    free(visited);
                    return NULL;
                }
            }
        }
    }

    free(component_pixels);

    free(visited);
    printf("Found %d contours\n", contours->count);
    return contours;
}

void freeContours(Contours *contours)
{
    if (!contours)
        return;

    for (int i = 0; i < contours->count; i++)
    {
        free(contours->contours[i].points);
    }
    free(contours->contours);
    free(contours);
}

/*
    Calculate the arc length (perimeter) of a contour.
    Resource: https://en.wikipedia.org/wiki/Contour_(computer_vision)
*/
double arcLength(const Contour *contour, int closed)
{
    if (!contour || contour->count < 2)
    {
        return 0.0;
    }

    double length = 0.0;

    // Calculate distance between consecutive points
    for (int i = 0; i < contour->count - 1; i++)
    {
        Point p1 = contour->points[i];
        Point p2 = contour->points[i + 1];
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        length += sqrt(dx * dx + dy * dy);
    }

    // If closed, add distance from last point back to first point
    if (closed && contour->count > 2)
    {
        Point p1 = contour->points[contour->count - 1];
        Point p2 = contour->points[0];
        double dx = p2.x - p1.x;
        double dy = p2.y - p1.y;
        length += sqrt(dx * dx + dy * dy);
    }

    return length;
}

/*
    Filter contours based on minimum arc length.
*/
Contours *filterContoursByLength(const Contours *contours, double min_length)
{
    if (!contours)
    {
        return NULL;
    }

    Contours *filtered = malloc(sizeof(Contours));
    if (!filtered)
    {
        fprintf(stderr,
                "Error: Failed to allocate filtered contours structure\n");
        return NULL;
    }

    filtered->contours = NULL;
    filtered->count = 0;
    filtered->capacity = 0;

    for (int i = 0; i < contours->count; i++)
    {
        const Contour *contour = &contours->contours[i];

        double length = arcLength(contour, 0);

        if (length > min_length)
        {
            Contour new_contour;
            new_contour.count = contour->count;
            new_contour.capacity = contour->count;
            new_contour.points = malloc(contour->count * sizeof(Point));

            if (!new_contour.points)
            {
                fprintf(
                    stderr,
                    "Error: Failed to allocate points for filtered contour\n");
                freeContours(filtered);
                return NULL;
            }

            memcpy(new_contour.points, contour->points,
                   contour->count * sizeof(Point));

            if (!addContourToContours(filtered, new_contour))
            {
                free(new_contour.points);
                freeContours(filtered);
                return NULL;
            }
        }else{
            // printf("Line of length %f not added\n", length);
        }
    }

    printf("Filtered contours: %d -> %d (min_length=%.2f)\n", contours->count,
           filtered->count, min_length);
    return filtered;
}

/*
    Calculate the bounding rectangle of a contour.
*/
int boundingRect(const Contour *contour, Rect *rect)
{
    if (!contour || contour->count == 0)
    {
        return 0;
    }

    int min_x = contour->points[0].x;
    int max_x = contour->points[0].x;
    int min_y = contour->points[0].y;
    int max_y = contour->points[0].y;

    for (int i = 1; i < contour->count; i++)
    {
        Point p = contour->points[i];
        if (p.x < min_x)
            min_x = p.x;
        if (p.x > max_x)
            max_x = p.x;
        if (p.y < min_y)
            min_y = p.y;
        if (p.y > max_y)
            max_y = p.y;
    }

    rect->x = min_x;
    rect->y = min_y;
    rect->width = max_x - min_x + 1;
    rect->height = max_y - min_y + 1;

    return 1;
}

/*
    Find the contour with maximum area (approximated by bounding rectangle
   area).
*/
int findMaxAreaContour(const Contours *contours)
{
    if (!contours || contours->count == 0)
    {
        return -1;
    }

    int max_index = 0;
    int max_area = 0;

    for (int i = 0; i < contours->count; i++)
    {
        Rect rect;
        if (boundingRect(&contours->contours[i], &rect))
        {
            int area = rect.width * rect.height;
            if (area > max_area)
            {
                max_area = area;
                max_index = i;
            }
        }
    }

    return max_index;
}

/*
    Find the bounding rectangle that contains all given rectangles with padding.
*/
int getBoundingRectOfRects(const Rect *rects, int count, int padding,
                           int image_width, int image_height, Rect *result)
{
    if (!rects || count <= 0 || !result)
    {
        return 0;
    }

    int min_x = rects[0].x;
    int min_y = rects[0].y;
    int max_x = rects[0].x + rects[0].width;
    int max_y = rects[0].y + rects[0].height;

    for (int i = 1; i < count; i++)
    {
        int x1 = rects[i].x;
        int y1 = rects[i].y;
        int x2 = x1 + rects[i].width;
        int y2 = y1 + rects[i].height;

        if (x1 < min_x)
            min_x = x1;
        if (y1 < min_y)
            min_y = y1;
        if (x2 > max_x)
            max_x = x2;
        if (y2 > max_y)
            max_y = y2;
    }

    result->x = (min_x - padding > 0) ? min_x - padding : 0;
    result->y = (min_y - padding > 0) ? min_y - padding : 0;
    int width_calc = max_x - min_x + 2 * padding;
    result->width = (width_calc < image_width - result->x)
                        ? width_calc
                        : image_width - result->x;
    int height_calc = max_y - min_y + 2 * padding;
    result->height = (height_calc < image_height - result->y)
                         ? height_calc
                         : image_height - result->y;

    return 1;
}

/*
    Estimate noise level in a grayscale image using local variance analysis.
    Returns a value between 0.0 (no noise) and 1.0 (high noise).
*/
double estimate_noise_level(Image *image)
{
    if (!image || !image->gray_pixels)
    {
        return 0.0;
    }

    double total_variance = 0.0;
    int sample_count = 0;
    int step = 10;

    for (int y = step; y < image->height - step; y += step)
    {
        for (int x = step; x < image->width - step; x += step)
        {
            // Calculate local variance in a small window
            double local_mean = 0.0;
            double local_var = 0.0;
            int window_size = 3;
            int count = 0;

            for (int wy = -window_size / 2; wy <= window_size / 2; wy++)
            {
                for (int wx = -window_size / 2; wx <= window_size / 2; wx++)
                {
                    int px = x + wx;
                    int py = y + wy;
                    if (px >= 0 && px < image->width && py >= 0 &&
                        py < image->height)
                    {
                        uint8_t val =
                            image->gray_pixels[py * image->width + px];
                        local_mean += val;
                        count++;
                    }
                }
            }

            if (count > 0)
            {
                local_mean /= count;

                // Calculate variance
                for (int wy = -window_size / 2; wy <= window_size / 2; wy++)
                {
                    for (int wx = -window_size / 2; wx <= window_size / 2; wx++)
                    {
                        int px = x + wx;
                        int py = y + wy;
                        if (px >= 0 && px < image->width && py >= 0 &&
                            py < image->height)
                        {
                            uint8_t val =
                                image->gray_pixels[py * image->width + px];
                            double diff = val - local_mean;
                            local_var += diff * diff;
                        }
                    }
                }

                local_var /= count;
                total_variance += local_var;
                sample_count++;
            }
        }
    }

    if (sample_count == 0)
        return 0.0;

    total_variance /= sample_count;

    double noise_level = total_variance / 1000.0;
    return (noise_level > 1.0) ? 1.0 : noise_level;
}

/*
    Apply adaptive denoising based on estimated noise level.
    Uses lighter denoising to preserve thin lines while reducing noise.
*/
void adaptive_denoise(Image *image)
{
    if (!image || !image->gray_pixels)
    {
        fprintf(stderr, "Error: Invalid image for adaptive_denoise\n");
        return;
    }

    double noise_level = estimate_noise_level(image);
    printf("Estimated noise level: %.3f\n", noise_level);
    printf("LEVEl IS: %f\n", noise_level);
    if (noise_level < 0.1 || (noise_level < 0.8 && noise_level > 0.7))
    {
        gaussian_blur(image, 3, 0.5);
        printf("Applied light denoising (noise_level < 0.1)\n");
    }
    else if (noise_level < 0.3)
    {
        gaussian_blur(image, 3, 1.0);
        printf("Applied moderate denoising (0.1 <= noise_level < 0.3)\n");
    }
    else
    {
        gaussian_blur(image, 5, 1.5);
        printf("Applied stronger denoising (noise_level >= 0.3)\n");
    }
}

/*
    Apply adaptive morphological cleaning that preserves thin lines.
    Uses smaller kernels and minimal operations to avoid removing thin features.
*/
void adaptive_morphological_clean(Image *image)
{
    if (!image || !image->gray_pixels)
    {
        fprintf(stderr,
                "Error: Invalid image for adaptive_morphological_clean\n");
        return;
    }

    double noise_level = estimate_noise_level(image);

    if (noise_level > 0.05)
    {
        StructuringElement *kernel = getStructuringElement(MORPH_CROSS, 2, 2);
        if (kernel)
        {
            morphologyEx(image, MORPH_OPEN, kernel, 1);
            freeStructuringElement(kernel);
            printf("Applied morphological opening for noise reduction (kernel "
                   "2x2 cross)\n");
        }
    }
    else
    {
        printf("Skipped morphological cleaning (noise level too low)\n");
    }
}

/*
    Rotate an image by a given angle in degrees.
    Creates a square output filled with white background.
    !Warning: Modifies the image in place. The image becomes square.
*/
void rotate_image(Image *image, double angle)
{
    if (!image)
    {
        fprintf(stderr, "Error: Invalid image for rotation\n");
        return;
    }

    if (angle == 0.0)
    {
        printf("No rotation needed (angle = 0)\n");
        return;
    }

    while (angle > 180.0) angle -= 360.0;
    while (angle < -180.0) angle += 360.0;

    double radians = angle * M_PI / 180.0;

    int original_width = image->width;
    int original_height = image->height;

    double cos_angle = fabs(cos(radians));
    double sin_angle = fabs(sin(radians));

    int rotated_width = (int)ceil(original_width * cos_angle + original_height * sin_angle);
    int rotated_height = (int)ceil(original_width * sin_angle + original_height * cos_angle);

    int new_width = rotated_width - 1;
    int new_height = rotated_height - 1;

    if (new_width <= 0 || new_height <= 0)
    {
        fprintf(stderr, "Error: Invalid dimensions after rotation (%dx%d)\n", new_width, new_height);
        return;
    }

    cairo_surface_t *source_surface = NULL;
    cairo_surface_t *target_surface = NULL;
    cairo_t *cr = NULL;

    if (!image->rgba_pixels)
    {
        fprintf(stderr, "Error: Rotation only supports RGB images\n");
        return;
    }

    uint32_t *argb_pixels = (uint32_t *)malloc(original_width * original_height * sizeof(uint32_t));
    if (!argb_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate ARGB buffer for Cairo\n");
        return;
    }

    for (int i = 0; i < original_width * original_height; i++)
    {
        uint32_t rgba = image->rgba_pixels[i];
        uint8_t r = (rgba >> 24) & 0xFF;
        uint8_t g = (rgba >> 16) & 0xFF;
        uint8_t b = (rgba >> 8) & 0xFF;
        uint8_t a = rgba & 0xFF;
        argb_pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    source_surface = cairo_image_surface_create_for_data(
        (unsigned char *)argb_pixels, CAIRO_FORMAT_ARGB32,
        original_width, original_height, original_width * 4);
    free(argb_pixels);

    if (!source_surface || cairo_surface_status(source_surface) != CAIRO_STATUS_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create Cairo source surface\n");
        return;
    }

    target_surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, new_width, new_height);
    if (!target_surface || cairo_surface_status(target_surface) != CAIRO_STATUS_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create Cairo target surface\n");
        cairo_surface_destroy(source_surface);
        return;
    }

    cr = cairo_create(target_surface);
    if (!cr || cairo_status(cr) != CAIRO_STATUS_SUCCESS)
    {
        fprintf(stderr, "Error: Failed to create Cairo context\n");
        cairo_surface_destroy(source_surface);
        cairo_surface_destroy(target_surface);
        return;
    }

    cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 1.0);
    cairo_paint(cr);

    cairo_translate(cr, new_width / 2.0, new_height / 2.0);
    cairo_rotate(cr, radians);
    cairo_translate(cr, -original_width / 2.0, -original_height / 2.0);

    cairo_pattern_t *pattern = cairo_pattern_create_for_surface(source_surface);
    cairo_pattern_set_filter(pattern, CAIRO_FILTER_BILINEAR);
    cairo_set_source(cr, pattern);

    cairo_paint(cr);

    cairo_pattern_destroy(pattern);
    cairo_destroy(cr);
    cairo_surface_destroy(source_surface);

    unsigned char *target_data = cairo_image_surface_get_data(target_surface);
    int target_stride = cairo_image_surface_get_stride(target_surface);

    uint32_t *new_rgba_pixels = (uint32_t *)malloc(new_width * new_height * sizeof(uint32_t));
    if (!new_rgba_pixels)
    {
        fprintf(stderr, "Error: Failed to allocate new RGBA buffer\n");
        cairo_surface_destroy(target_surface);
        return;
    }

    for (int y = 0; y < new_height; y++)
    {
        for (int x = 0; x < new_width; x++)
        {
            int idx = y * new_width + x;
            uint32_t *pixel_ptr = (uint32_t *)(target_data + y * target_stride + x * 4);
            uint32_t argb = *pixel_ptr;

            uint8_t a = (argb >> 24) & 0xFF;
            uint8_t r = (argb >> 16) & 0xFF;
            uint8_t g = (argb >> 8) & 0xFF;
            uint8_t b = argb & 0xFF;

            new_rgba_pixels[idx] = (r << 24) | (g << 16) | (b << 8) | a;
        }
    }

    free(image->rgba_pixels);
    free(image->gray_pixels);

    image->rgba_pixels = new_rgba_pixels;
    image->gray_pixels = NULL;
    image->is_grayscale = false;
    image->width = new_width;
    image->height = new_height;

    cairo_surface_destroy(target_surface);

    printf("Rotated image by %.2f degrees (%dx%d -> %dx%d, minimal white padding)\n",
           angle, original_width, original_height, new_width, new_height);
}