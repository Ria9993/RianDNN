#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.01f;
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 100;
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(1, "None"); //output layer
	
	random_device rd;
	mt19937 rnd(rd());
	uniform_int_distribution <> rd100(0, 100);
	for (int i = 0; i < 1000; i++) {
		double input[110];
		for (int j = 0; j < 100; j++)
			input[j] = rd100(rnd);
		double* output = dnn.Forward(input);
		printf("%4d : %lf\n", i,*output);
	}
	return 0;
}