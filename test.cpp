#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.01f;
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 10; //input Layer
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(2, "None");
	//dnn.AddLayer(2, "Softmax"); //output layer 

	/*DEBUG*/
	random_device rd;
	mt19937 rnd(rd());
	uniform_int_distribution <> rd100(0, 100);
	for (int i = 0; i < 1000; i++) {
		double input[110];
		for (int j = 0; j < 100; j++) {
			//input[j] = rd100(rnd);
			input[j] = 0.1f;
		}
		double* output = dnn.Forward(input);
		double target[10] = { 2.0f, 4.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
		dnn.Optimize(target);
		printf("weights and bias\n");
		for (int j = 0; j < dnn.layer_[dnn.layer_num_ - 1].node_num_; j++) {
			for (int k = 0; k < dnn.layer_[dnn.layer_num_ - 1].last_node_num_; k++) {
				printf("%8.5lf(%8.5lf) ", dnn.layer_[dnn.layer_num_ - 1].weight_[j][k], dnn.layer_[dnn.layer_num_ - 1].weight_grad_[j][k]);
			}
			printf(" Bias : %8.5lf  Grad : %8.5lf\n", dnn.layer_[dnn.layer_num_ - 1].bias_[j],dnn.layer_[dnn.layer_num_-1].grad_[j]);
		}
		printf("outputs\n");
		printf("%7.4lf  %7.4lf   loss : %8.5lf\n", *output, output[1], dnn.loss_);
		dnn.GradZero();
		scanf("%*c"); //pause
	}
	return 0;
}