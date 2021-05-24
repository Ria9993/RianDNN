#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.05f;
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 1; //input Layer
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(2, "Softmax");
	//dnn.AddLayer(2, "Softmax"); //output layer 

	/*DEBUG*/
	random_device rd;
	mt19937 rnd(rd());
	uniform_real_distribution <double> rd100(-1, 1);
	for (int i = 0; i < 10000; i++) {
		double input[110];
		for (int j = 0; j < 100; j++) {
			//input[j] = rd100(rnd);
			input[j] = 0.5f;
		}
		double target[2] = { 0.3f, 0.7f};
		double* output = dnn.Forward(input,target);
		if (i % 2 == 0) {
			dnn.Optimize(target);
		}
	}
	return 0;
}