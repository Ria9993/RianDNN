#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.005f;
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 1; //input Layer
	dnn.AddLayer(30, "None");
	dnn.AddLayer(30, "None");
	dnn.AddLayer(30, "None");
	dnn.AddLayer(2, "Softmax");
	//dnn.AddLayer(2, "Softmax"); //output layer 

	/*DEBUG*/
	random_device rd;
	mt19937 rnd(rd());
	uniform_real_distribution <double> rd100(-1, 1);
	for (int i = 0; i < 1000000; i++) {
		double input[110];
		for (int j = 0; j < 100; j++) {
			//input[j] = rd100(rnd);
			input[j] = 0.5f;
		}
		double target[10] = { 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
		double* output = dnn.Forward(input,target);
		//target[0] = input[0] * 2;
		if(i%2==0)
			dnn.Optimize(target);
		printf("weights and bias\n");
		for (int j = 0; j < dnn.layer_[dnn.layer_num_ - 1].node_num_; j++) {
			for (int k = 0; k < dnn.layer_[dnn.layer_num_ - 1].last_node_num_; k++) {
				printf("%8.5lf(%8.5lf) ", dnn.layer_[dnn.layer_num_ - 1].weight_[j][k], dnn.layer_[dnn.layer_num_ - 1].weight_grad_[j][k]);
			}
			printf(" Bias : %8.5lf  Grad : %8.5lf\n", dnn.layer_[dnn.layer_num_ - 1].bias_[j],dnn.layer_[dnn.layer_num_-1].grad_[j]);
		}
		printf("input : %5.2lf target : %5.2lf\n", input[0],target[0]);
		printf("outputs\n");
		printf("%7.4lf  %7.4lf   loss : %8.5lf\n", *output, output[1], dnn.loss_);
		dnn.GradZero();
		//if(i>10000)
			scanf("%*c"); //pause
	}
	return 0;
}