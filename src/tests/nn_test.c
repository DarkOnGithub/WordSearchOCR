#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double y) { return y * (1 - y); }
double random_nb_gen() { return (2 * rand()) / ((double)RAND_MAX - 1); }

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
    printf("\n=== Neural Network Test Summary ===\n");
    printf("Total tests: %d\n", tests_run);
    printf(GREEN "Passed: %d" RESET "\n", tests_passed);
    printf(RED "Failed: %d" RESET "\n", tests_failed);
    printf("Success rate: %.1f%%\n", tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0);
}

bool test_sigmoid(void) {
    test_start("Sigmoid function");

    double result1 = sigmoid(0.0);
    double result2 = sigmoid(1.0);
    double result3 = sigmoid(-1.0);

    bool pass = (fabs(result1 - 0.5) < 1e-10) &&
                (fabs(result2 - 0.7310585786) < 1e-5) &&
                (fabs(result3 - 0.2689414214) < 1e-5) &&
                (sigmoid(100.0) > 0.99) &&
                (sigmoid(-100.0) < 0.01);

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Sigmoid function incorrect");
        return false;
    }
}

bool test_sigmoid_derivative(void) {
    test_start("Sigmoid derivative");

    double y = sigmoid(0.0);
    double deriv = sigmoid_derivative(y);

    bool pass = fabs(deriv - 0.25) < 1e-10;

    for (double x = -5.0; x <= 5.0; x += 0.5) {
        double s = sigmoid(x);
        double d = sigmoid_derivative(s);
        if (d < 0 || d > 0.25) {
            pass = false;
            break;
        }
    }

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Sigmoid derivative incorrect");
        return false;
    }
}

bool test_random_range(void) {
    test_start("Random number generator range");

    srand(42);
    bool pass = true;

    for (int i = 0; i < 1000; i++) {
        double r = random_nb_gen();
        if (r < -1.0 || r > 1.0) {
            pass = false;
            break;
        }
    }

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Random number out of range");
        return false;
    }
}

bool test_random_seed(void) {
    test_start("Random number generator seed reproducibility");

    srand(42);
    double r1 = random_nb_gen();
    double r2 = random_nb_gen();

    srand(42);
    double r1_repeat = random_nb_gen();
    double r2_repeat = random_nb_gen();

    bool pass = (fabs(r1 - r1_repeat) < 1e-10) &&
                 (fabs(r2 - r2_repeat) < 1e-10);

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Random seed not reproducible");
        return false;
    }
}

bool test_forward_pass(void) {
    test_start("Neural network forward pass");

    double w1[2][2] = {{0.5, 0.3}, {0.2, 0.4}};
    double b1[2] = {0.1, 0.2};
    double w2[2] = {0.6, 0.7};
    double b2 = 0.3;

    double inputs[2] = {0.0, 1.0};

    double h0 = sigmoid(inputs[0] * w1[0][0] + inputs[1] * w1[1][0] + b1[0]);
    double h1 = sigmoid(inputs[0] * w1[0][1] + inputs[1] * w1[1][1] + b1[1]);

    double output = sigmoid(h0 * w2[0] + h1 * w2[1] + b2);

    bool pass = (output >= 0.0) && (output <= 1.0) &&
                !isnan(output) && !isinf(output);

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Forward pass produces invalid output");
        return false;
    }
}

bool test_xnor_truth_table(void) {
    test_start("XNOR truth table");

    double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    double expected[4] = {1.0, 0.0, 0.0, 1.0};

    bool pass = true;
    for (int i = 0; i < 4; i++) {
        bool a = inputs[i][0] != 0.0;
        bool b = inputs[i][1] != 0.0;
        double expected_value = (a == b) ? 1.0 : 0.0;

        if (fabs(expected[i] - expected_value) > 1e-10) {
            pass = false;
            break;
        }
    }

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("XNOR truth table incorrect");
        return false;
    }
}

bool test_neural_network_training(void) {
    test_start("Neural network training");

    double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    double outputs[4] = {1.0, 0.0, 0.0, 1.0};

    double w1[2][2];
    double b1[2];
    double w2[2];
    double b2;

    srand(42);

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            w1[i][j] = random_nb_gen();
        }
    }
    for (int j = 0; j < 2; ++j)
        b1[j] = random_nb_gen();
    for (int j = 0; j < 2; ++j)
        w2[j] = random_nb_gen();
    b2 = random_nb_gen();

    double lr = 0.5;
    int epochs = 10000;

    double h[4][2];
    double o[4];
    double delta_o[4];
    double delta_h[4][2];

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int s = 0; s < 4; ++s) {
            for (int j = 0; j < 2; ++j) {
                double z = inputs[s][0] * w1[0][j] + inputs[s][1] * w1[1][j] + b1[j];
                h[s][j] = sigmoid(z);
            }
            double z_out = h[s][0] * w2[0] + h[s][1] * w2[1] + b2;
            o[s] = sigmoid(z_out);
        }

        for (int s = 0; s < 4; ++s) {
            double err = outputs[s] - o[s];
            delta_o[s] = err * sigmoid_derivative(o[s]);
            for (int j = 0; j < 2; ++j) {
                delta_h[s][j] = delta_o[s] * w2[j] * sigmoid_derivative(h[s][j]);
            }
        }

        double dw2_0 = 0.0, dw2_1 = 0.0, db2 = 0.0;
        for (int s = 0; s < 4; ++s) {
            dw2_0 += h[s][0] * delta_o[s];
            dw2_1 += h[s][1] * delta_o[s];
            db2 += delta_o[s];
        }
        w2[0] += dw2_0 * lr;
        w2[1] += dw2_1 * lr;
        b2 += db2 * lr;

        double dw1_00 = 0.0, dw1_10 = 0.0;
        double dw1_01 = 0.0, dw1_11 = 0.0;
        double db1_0 = 0.0, db1_1 = 0.0;
        for (int s = 0; s < 4; ++s) {
            dw1_00 += inputs[s][0] * delta_h[s][0];
            dw1_01 += inputs[s][0] * delta_h[s][1];
            dw1_10 += inputs[s][1] * delta_h[s][0];
            dw1_11 += inputs[s][1] * delta_h[s][1];
            db1_0 += delta_h[s][0];
            db1_1 += delta_h[s][1];
        }
        w1[0][0] += dw1_00 * lr;
        w1[1][0] += dw1_10 * lr;
        w1[0][1] += dw1_01 * lr;
        w1[1][1] += dw1_11 * lr;
        b1[0] += db1_0 * lr;
        b1[1] += db1_1 * lr;
    }

    int correct = 0;
    double total_error = 0.0;
    for (int s = 0; s < 4; ++s) {
        double hh0 = sigmoid(inputs[s][0] * w1[0][0] + inputs[s][1] * w1[1][0] + b1[0]);
        double hh1 = sigmoid(inputs[s][0] * w1[0][1] + inputs[s][1] * w1[1][1] + b1[1]);
        double out = sigmoid(hh0 * w2[0] + hh1 * w2[1] + b2);
        int rounded = (int)round(out);
        if (rounded == (int)outputs[s])
            correct++;
        total_error += pow(outputs[s] - out, 2);
    }

    double accuracy = (correct / 4.0) * 100.0;
    double mse = total_error / 4.0;

    bool pass = (accuracy >= 75.0) && (mse >= 0.0) && (mse < 1.0);

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Training accuracy too low");
        return false;
    }
}

bool test_weight_initialization(void) {
    test_start("Weight initialization");

    srand(42);
    double w1[2][2];
    double b1[2];
    double w2[2];
    double b2;

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            w1[i][j] = random_nb_gen();
        }
    }
    for (int j = 0; j < 2; ++j)
        b1[j] = random_nb_gen();
    for (int j = 0; j < 2; ++j)
        w2[j] = random_nb_gen();
    b2 = random_nb_gen();

    bool pass = true;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            if (w1[i][j] < -1.0 || w1[i][j] > 1.0) {
                pass = false;
                break;
            }
        }
        if (b1[i] < -1.0 || b1[i] > 1.0) {
            pass = false;
            break;
        }
        if (w2[i] < -1.0 || w2[i] > 1.0) {
            pass = false;
            break;
        }
    }
    if (b2 < -1.0 || b2 > 1.0) {
        pass = false;
    }

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Weights out of range");
        return false;
    }
}

bool test_gradient_computation(void) {
    test_start("Gradient computation");

    double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    double outputs[4] = {1.0, 0.0, 0.0, 1.0};

    double w1[2][2] = {{0.5, 0.3}, {0.2, 0.4}};
    double b1[2] = {0.1, 0.2};
    double w2[2] = {0.6, 0.7};
    double b2 = 0.3;

    double h[4][2];
    double o[4];
    double delta_o[4];
    double delta_h[4][2];

    for (int s = 0; s < 4; ++s) {
        for (int j = 0; j < 2; ++j) {
            double z = inputs[s][0] * w1[0][j] + inputs[s][1] * w1[1][j] + b1[j];
            h[s][j] = sigmoid(z);
        }
        double z_out = h[s][0] * w2[0] + h[s][1] * w2[1] + b2;
        o[s] = sigmoid(z_out);
    }

    for (int s = 0; s < 4; ++s) {
        double err = outputs[s] - o[s];
        delta_o[s] = err * sigmoid_derivative(o[s]);
        for (int j = 0; j < 2; ++j) {
            delta_h[s][j] = delta_o[s] * w2[j] * sigmoid_derivative(h[s][j]);
        }
    }

    bool pass = true;
    for (int s = 0; s < 4; ++s) {
        if (!isfinite(delta_o[s])) {
            pass = false;
            break;
        }
        for (int j = 0; j < 2; ++j) {
            if (!isfinite(delta_h[s][j])) {
                pass = false;
                break;
            }
        }
    }

    if (pass) {
        test_pass();
        return true;
    } else {
        test_fail("Gradients contain NaN or Inf");
        return false;
    }
}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    printf("=== Neural Network (XNOR) Test Suite ===\n\n");

    test_sigmoid();
    test_sigmoid_derivative();
    test_random_range();
    test_random_seed();
    test_forward_pass();
    test_xnor_truth_table();
    test_weight_initialization();
    test_gradient_computation();
    test_neural_network_training();

    test_summary();

    return (tests_failed > 0) ? 1 : 0;
}

