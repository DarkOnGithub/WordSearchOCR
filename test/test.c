#include <stdio.h>
#include "nn/core/tensor.h"
#include "nn/core/init.h"

int main() {
    printf("Testing tensor printing with variable dimensions\n\n");

    // Test 0D tensor (scalar)
    printf("0D tensor:\n");
    int scalar_shape[] = {};
    Tensor* scalar = tensor_create_ones(scalar_shape, 0);
    tensor_print(scalar);
    tensor_free(scalar);
    printf("\n");

    // Test 1D tensor
    printf("1D tensor:\n");
    int shape1d[] = {5};
    Tensor* tensor1d = tensor_create_random(shape1d, 1);
    tensor_print(tensor1d);
    tensor_free(tensor1d);
    printf("\n");

    // Test 2D tensor
    printf("2D tensor:\n");
    int shape2d[] = {3, 4};
    Tensor* tensor2d = tensor_create_random(shape2d, 2);
    tensor_print(tensor2d);
    tensor_free(tensor2d);
    printf("\n");

    // Test 3D tensor
    printf("3D tensor:\n");
    int shape3d[] = {2, 3, 4};
    Tensor* tensor3d = tensor_create_random(shape3d, 3);
    tensor_print(tensor3d);
    tensor_free(tensor3d);
    printf("\n");

    // Test 4D tensor
    printf("4D tensor:\n");
    int shape4d[] = {2, 2, 3, 3};
    Tensor* tensor4d = tensor_create_random(shape4d, 4);
    tensor_print(tensor4d);
    tensor_free(tensor4d);
    printf("\n");

    // Test large tensor (should truncate)
    printf("Large tensor (should show truncation):\n");
    int shape_large[] = {10, 8};
    Tensor* tensor_large = tensor_create_random(shape_large, 2);
    tensor_print(tensor_large);
    tensor_free(tensor_large);
    printf("\n");

    printf("All tests completed!\n");

    printf("\nTesting neural network weight initialization functions\n\n");

    // Test Xavier uniform initialization
    printf("Xavier uniform initialization (2x3 matrix):\n");
    int weight_shape[] = {2, 3}; // out_features x in_features
    Tensor* xavier_uniform = tensor_create(weight_shape, 2);
    init_xavier_uniform(xavier_uniform);
    tensor_print(xavier_uniform);
    tensor_free(xavier_uniform);
    printf("\n");

    // Test Xavier normal initialization
    printf("Xavier normal initialization (2x3 matrix):\n");
    Tensor* xavier_normal = tensor_create(weight_shape, 2);
    init_xavier_normal(xavier_normal);
    tensor_print(xavier_normal);
    tensor_free(xavier_normal);
    printf("\n");

    // Test Kaiming uniform initialization
    printf("Kaiming uniform initialization (2x3 matrix):\n");
    Tensor* kaiming_uniform = tensor_create(weight_shape, 2);
    init_kaiming_uniform(kaiming_uniform);
    tensor_print(kaiming_uniform);
    tensor_free(kaiming_uniform);
    printf("\n");

    // Test Kaiming normal initialization
    printf("Kaiming normal initialization (2x3 matrix):\n");
    Tensor* kaiming_normal = tensor_create(weight_shape, 2);
    init_kaiming_normal(kaiming_normal);
    tensor_print(kaiming_normal);
    tensor_free(kaiming_normal);
    printf("\n");

    printf("Initialization tests completed!\n");

    printf("\nTesting tensor_matmul function (PyTorch-style)\n\n");

    // Test 1D x 1D (dot product -> scalar)
    printf("1D x 1D dot product:\n");
    int vec_shape[] = {3};
    Tensor* vec1 = tensor_create_random(vec_shape, 1);
    Tensor* vec2 = tensor_create_random(vec_shape, 1);

    printf("Vector A (3):\n");
    tensor_print(vec1);
    printf("Vector B (3):\n");
    tensor_print(vec2);

    Tensor* dot_result = tensor_matmul(vec1, vec2);
    if (dot_result) {
        printf("Result A @ B (scalar):\n");
        tensor_print(dot_result);
        tensor_free(dot_result);
    } else {
        printf("Dot product failed!\n");
    }

    tensor_free(vec1);
    tensor_free(vec2);
    printf("\n");

    // Test 1D x 2D (vector-matrix multiplication)
    printf("1D x 2D vector-matrix multiplication:\n");
    int vec_shape2[] = {3};
    int mat_shape[] = {3, 4};
    Tensor* vec = tensor_create_random(vec_shape2, 1);
    Tensor* mat = tensor_create_random(mat_shape, 2);

    printf("Vector A (3):\n");
    tensor_print(vec);
    printf("Matrix B (3x4):\n");
    tensor_print(mat);

    Tensor* vec_mat_result = tensor_matmul(vec, mat);
    if (vec_mat_result) {
        printf("Result A @ B (4):\n");
        tensor_print(vec_mat_result);
        tensor_free(vec_mat_result);
    } else {
        printf("Vector-matrix multiplication failed!\n");
    }

    tensor_free(vec);
    tensor_free(mat);
    printf("\n");

    // Test 2D x 1D (matrix-vector multiplication)
    printf("2D x 1D matrix-vector multiplication:\n");
    int mat_shape2[] = {2, 3};
    int vec_shape3[] = {3};
    Tensor* mat2 = tensor_create_random(mat_shape2, 2);
    Tensor* vec3 = tensor_create_random(vec_shape3, 1);

    printf("Matrix A (2x3):\n");
    tensor_print(mat2);
    printf("Vector B (3):\n");
    tensor_print(vec3);

    Tensor* mat_vec_result = tensor_matmul(mat2, vec3);
    if (mat_vec_result) {
        printf("Result A @ B (2):\n");
        tensor_print(mat_vec_result);
        tensor_free(mat_vec_result);
    } else {
        printf("Matrix-vector multiplication failed!\n");
    }

    tensor_free(mat2);
    tensor_free(vec3);
    printf("\n");

    // Test 2D x 2D matrix multiplication
    printf("2D x 2D matrix multiplication:\n");
    int a_shape[] = {2, 3};
    int b_shape[] = {3, 4};
    Tensor* a = tensor_create_random(a_shape, 2);
    Tensor* b = tensor_create_random(b_shape, 2);

    printf("Matrix A (2x3):\n");
    tensor_print(a);
    printf("Matrix B (3x4):\n");
    tensor_print(b);

    Tensor* c = tensor_matmul(a, b);
    if (c) {
        printf("Result A @ B (2x4):\n");
        tensor_print(c);
        tensor_free(c);
    } else {
        printf("Matrix multiplication failed!\n");
    }

    tensor_free(a);
    tensor_free(b);
    printf("\n");

    // Test 3D x 3D batch matrix multiplication
    printf("3D x 3D batch matrix multiplication:\n");
    int a3d_shape[] = {2, 3, 4};  // batch=2, m=3, k=4
    int b3d_shape[] = {2, 4, 5};  // batch=2, k=4, n=5
    Tensor* a3d = tensor_create_random(a3d_shape, 3);
    Tensor* b3d = tensor_create_random(b3d_shape, 3);

    printf("Tensor A (2x3x4):\n");
    tensor_print(a3d);
    printf("Tensor B (2x4x5):\n");
    tensor_print(b3d);

    Tensor* c_batch = tensor_matmul(a3d, b3d);
    if (c_batch) {
        printf("Result A @ B (2x3x5):\n");
        tensor_print(c_batch);
        tensor_free(c_batch);
    } else {
        printf("Batch matrix multiplication failed!\n");
    }

    tensor_free(a3d);
    tensor_free(b3d);
    printf("\n");

    // Test 3D x 2D batch matrix multiplication with broadcasting
    printf("3D x 2D batch matrix multiplication (broadcasting):\n");
    int a3d_shape2[] = {2, 3, 4};  // batch=2, m=3, k=4
    int b2d_shape[] = {4, 5};     // k=4, n=5
    Tensor* a3d2 = tensor_create_random(a3d_shape2, 3);
    Tensor* b2d = tensor_create_random(b2d_shape, 2);

    printf("Tensor A (2x3x4):\n");
    tensor_print(a3d2);
    printf("Matrix B (4x5):\n");
    tensor_print(b2d);

    Tensor* c_batch2 = tensor_matmul(a3d2, b2d);
    if (c_batch2) {
        printf("Result A @ B (2x3x5):\n");
        tensor_print(c_batch2);
        tensor_free(c_batch2);
    } else {
        printf("Batch matrix multiplication with broadcasting failed!\n");
    }

    tensor_free(a3d2);
    tensor_free(b2d);
    printf("\n");

    printf("Matrix multiplication tests completed!\n");
    return 0;
}
