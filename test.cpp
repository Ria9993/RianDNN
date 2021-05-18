#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.input_num_ = 10;
	dnn.AddLayer(5, "ReLU");
	dnn.AddLayer(5, "ReLU");
	dnn.AddLayer(2, "ReLU"); //output layer
	return 0;
}