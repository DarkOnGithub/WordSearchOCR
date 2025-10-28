#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double y){
	return y * (1 - y);
}

double random_nb_gen(){
	return (2 * rand())/((double)RAND_MAX- 1);
}

int main(){
	double inputs[4][2] = {
        	{0.0, 0.0},
        	{0.0, 1.0},
        	{1.0, 0.0},
        	{1.0, 1.0}
	};
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

    	for (int s = 0; s < 4; ++s) {
        	double hh0 = sigmoid(inputs[s][0] * w1[0][0] + inputs[s][1] * w1[1][0] + b1[0]);
        	double hh1 = sigmoid(inputs[s][0] * w1[0][1] + inputs[s][1] * w1[1][1] + b1[1]);
        	double out = sigmoid(hh0 * w2[0] + hh1 * w2[1] + b2);
        	int rounded = (int) round(out);
        	printf("Input: [%g, %g] -> Output: %i (Output before roudning: %f)\n",inputs[s][0], inputs[s][1], rounded, out);
    	}

    	int correct = 0;
    	double total_error = 0.0;
    	for (int s = 0; s < 4; ++s) {
        	double hh0 = sigmoid(inputs[s][0] * w1[0][0] + inputs[s][1] * w1[1][0] + b1[0]);
        	double hh1 = sigmoid(inputs[s][0] * w1[0][1] + inputs[s][1] * w1[1][1] + b1[1]);
        	double out = sigmoid(hh0 * w2[0] + hh1 * w2[1] + b2);
        	int rounded = (int) round(out);
        	if (rounded == (int)outputs[s]) correct++;
        	total_error += pow(outputs[s] - out, 2);
    	}

    	printf("Accuracy: %d/4 (%.1f%%)\n", correct, (correct / 4.0) * 100.0);
    	printf("MSE: %.6f\n", total_error / 4.0);

    	printf("Hidden Layer Weights:\n");
    	printf("  w1[0][0] = %.6f, w1[0][1] = %.6f\n", w1[0][0], w1[0][1]);
    	printf("  w1[1][0] = %.6f, w1[1][1] = %.6f\n", w1[1][0], w1[1][1]);
    	printf("Hidden Layer Biases: b1[0] = %.6f, b1[1] = %.6f\n", b1[0], b1[1]);
    	printf("Output Layer Weights: w2[0] = %.6f, w2[1] = %.6f\n", w2[0], w2[1]);
    	printf("Output Layer Bias: b2 = %.6f\n", b2);

    return 0;
}

