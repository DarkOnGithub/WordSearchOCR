#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CBLAS_ROW_MAJOR 101
#define CBLAS_COL_MAJOR 102
#define CBLAS_NO_TRANS 111
#define CBLAS_TRANS 112
#define CBLAS_CONJ_TRANS 113

extern void cblas_dgemm(const int Order, const int TransA, const int TransB,
                        const int M, const int N, const int K,
                        const double alpha, const double *A, const int lda,
                        const double *B, const int ldb, const double beta,
                        double *C, const int ldc);

extern void cblas_daxpy(const int N, const double alpha, const double *X,
                        const int incX, double *Y, const int incY);

extern double cblas_ddot(const int N, const double *X, const int incX,
                         const double *Y, const int incY);

extern void cblas_dscal(const int N, const double alpha, double *X,
                        const int incX);

#ifdef __cplusplus
}
#endif

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void test_start(const char *test_name) {
    printf("Testing: %s... ", test_name);
    tests_run++;
}

void test_pass(void) {
    printf(GREEN "PASS" RESET "\n");
    tests_passed++;
}

void test_fail(const char *reason) {
    printf(RED "FAIL" RESET " - %s\n", reason);
    tests_failed++;
}

void test_summary(void) {
    printf("\n=== BLAS Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf(GREEN "Passed: %d" RESET "\n", tests_passed);
    printf(RED "Failed: %d" RESET "\n", tests_failed);
    printf("Success rate: %.1f%%\n", tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
}

bool test_blas_linked(void) {
    test_start("BLAS library linkage");

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {5.0, 6.0, 7.0, 8.0};

    cblas_daxpy(4, 1.0, x, 1, y, 1);

    test_pass();
    return true;
}

bool test_daxpy(void) {
    test_start("DAXPY (y = alpha*x + y)");

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {5.0, 6.0, 7.0, 8.0};
    double expected[4] = {6.0, 8.0, 10.0, 12.0};

    cblas_daxpy(4, 1.0, x, 1, y, 1);

    bool result = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(y[i] - expected[i]) > 1e-10) {
            result = false;
            break;
        }
    }

    if (result) {
        test_pass();
        return true;
    } else {
        test_fail("DAXPY result incorrect");
        return false;
    }
}

bool test_ddot(void) {
    test_start("DDOT (dot product)");

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {5.0, 6.0, 7.0, 8.0};
    double expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0;

    double result = cblas_ddot(4, x, 1, y, 1);

    if (fabs(result - expected) < 1e-10) {
        test_pass();
        return true;
    } else {
        test_fail("DDOT result incorrect");
        return false;
    }
}

bool test_dscal(void) {
    test_start("DSCAL (scale vector)");

    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double expected[4] = {2.0, 4.0, 6.0, 8.0};

    cblas_dscal(4, 2.0, x, 1);

    bool result = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(x[i] - expected[i]) > 1e-10) {
            result = false;
            break;
        }
    }

    if (result) {
        test_pass();
        return true;
    } else {
        test_fail("DSCAL result incorrect");
        return false;
    }
}

bool test_dgemm(void) {
    test_start("DGEMM (matrix multiplication)");

    double A[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double B[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double C[4] = {0.0, 0.0, 0.0, 0.0};

    double expected[4] = {22.0, 28.0, 49.0, 64.0};

    cblas_dgemm(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_NO_TRANS,
                2, 2, 3, 1.0, A, 3, B, 2, 0.0, C, 2);

    bool result = true;
    for (int i = 0; i < 4; i++) {
        if (fabs(C[i] - expected[i]) > 1e-10) {
            result = false;
            break;
        }
    }

    if (result) {
        test_pass();
        return true;
    } else {
        test_fail("DGEMM result incorrect");
        return false;
    }
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("=== BLAS Library Test Suite ===\n\n");

    test_blas_linked();
    test_daxpy();
    test_ddot();
    test_dscal();
    test_dgemm();

    test_summary();

    return (tests_failed > 0) ? 1 : 0;
}

