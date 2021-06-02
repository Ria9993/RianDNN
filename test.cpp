#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.05f;
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 2; //input Layer
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(30, "ReLU");
	dnn.AddLayer(1, "Softmax");
	//dnn.AddLayer(2, "Softmax"); //output layer 

	/*
	dnn.input_num_ = 10 * 10; //input Layer
	dnn.AddLayer("Conv2D", { 10, 10, 8, 3, 3, 1 });
	dnn.AddLayer("ReLU");
	dnn.AddLayer("BatchNorm");
	dnn.AddLayer("MaxPool", { 2, 2, 2 });
	dnn.AddLayer("Conv2D", { 5, 5, 16, 3, 3, 1 });
	dnn.AddLayer("ReLU");
	dnn.AddLayer("BatchNorm");
	dnn.AddLayer("MaxPool", { 2, 2, 2 });
	dnn.AddLayer("Flatten");
	dnn.AddLayer("Dense", { 55 });
	dnn.AddLayer("BatchNorm");
	dnn.AddLayer("ReLU");
	dnn.AddLayer("Dense", { 55 });
	dnn.AddLayer("BatchNorm");
	dnn.AddLayer("ReLU");
	dnn.AddLayer("Dense", { 10 });
	dnn.AddLayer("Softmax");
	*/
	/*
	dnn.input_num_ = 10 * 10; //input Layer
	dnn.Conv2D(10, 10, 8, 3, 3, 1);
	dnn.BatchNorm();
	dnn.ReLU();
	dnn.MaxPool(2, 2, 2);
	dnn.Flatten();
	dnn.Dense(55);
	dnn.BatchNorn();
	dnn.ReLU();
	*/

	/*DEBUG*/
	random_device rd;
	mt19937 rnd(rd());
	uniform_real_distribution <double> rd100(-1, 1);
	for (int i = 0; i < 10000; i++) {
		double input[100] = {1,0,};
		for (int j = 0; j < 100; j++) {
			//input[j] = rd100(rnd);
			input[j] = 0.5f;
		}
		double target[2] = { 0.3f, 0.7f};
		double* output = dnn.Forward(input,target);
		if (i % 5 == 0) {
			dnn.Optimize();
		}
	}
	return 0;
}