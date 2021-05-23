#pragma warning(disable:4996) //visual studio _s(bof) warning disable

#include "RianDNN/RianDNN.h"
using namespace RianDNN;

#include <stdio.h>

int main() {
	RianDNN::DNN dnn;
	dnn.learning_rate = 0.01f; 
	//dnn.momentum_rate = 0.1;
	dnn.input_num_ = 1; //input Layer
	dnn.AddLayer(3, "ReLu");
	dnn.AddLayer(3, "ReLu");
	dnn.AddLayer(3, "ReLu");
	//dnn.AddLayer(300, "ReLU");
	dnn.AddLayer(1, "None");
	//dnn.AddLayer(2, "Softmax"); //output layer 

	/*DEBUG*/
	random_device rd;
	mt19937 rnd(rd());
	uniform_int_distribution <> rd100(-10, 10);
	for (int i = 0; i < 100000; i++) {
		double input[110];
		for (int j = 0; j < 100; j++) {
			//input[j] = rd100(rnd);
			input[j] = 1.0f;
		}
		//double target = input[0] * 2;
		double target[2] = { 0.5f,0.7f };
		double* output = dnn.Forward(input, target);
		printf("weights and bias\n");
		for (int n = 0; n < dnn.layer_num_; n++) {
			for (int j = 0; j < dnn.layer_[n].node_num_; j++) {
				for (int k = 0; k < dnn.layer_[n].last_node_num_; k++) {
					printf("%8.5lf(%8.5lf) ", dnn.layer_[n].weight_[j][k], dnn.layer_[n].weight_grad_[j][k]);
				}
				printf("\n");
			}
			printf("Grad   : ");
			for (int j = 0; j < dnn.layer_[n].node_num_; j++)
				printf("%8.5lf ", dnn.layer_[n].grad_[j]);
			printf("\nResult : ");
			for (int j = 0; j < dnn.layer_[n].node_num_; j++)
				printf("%8.5lf ", dnn.layer_[n].result_[j]);
			printf("\n");
		}
		printf("Step : %4d\n", i);
		printf("input\n %7.4lf\n", input[0]);
		printf("output   target\n");
		printf("%7.4lf  %7.4lf   loss : %8.20lf\n", *output, target[0], dnn.loss_);
		//double target[10] = { 2.0f, 4.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
		if (i % 1==0) { //mini-batch learning
			if (i == 1) {
				printf(".\n");
			}
			dnn.Optimize(target);
			printf("�ߡߡߡߡߡߡߡߡߡ�!!! !!! !!! OPTIMIZED !!! !!! !!!�ߡߡߡߡߡߡߡߡߡ�\n");
			printf("weights and bias\n");
			for (int n = 0; n < dnn.layer_num_; n++) {
				for (int j = 0; j < dnn.layer_[n].node_num_; j++) {
					for (int k = 0; k < dnn.layer_[n].last_node_num_; k++) {
						printf("%8.5lf(%8.5lf) ", dnn.layer_[n].weight_[j][k], dnn.layer_[n].weight_grad_[j][k]);
					}
					printf("\n");
				}
				printf("Grad   : ");
				for (int j = 0; j < dnn.layer_[n].node_num_; j++)
					printf("%8.5lf ",dnn.layer_[n].grad_[j]);
				printf("\nBack   : ");
				for (int j = 0; j < dnn.layer_[n].node_num_; j++)
					printf("%8.5lf ", dnn.layer_[n].back_pass_[j]);
				printf("\nResult : ");
				for (int j = 0; j < dnn.layer_[n].node_num_; j++) 
					printf("%8.5lf ", dnn.layer_[n].result_[j]);
				printf("\n");
			}
			printf("Step : %4d\n", i);
			printf("input\n %7.4lf\n", input[0]);
			printf("output   target\n");
			printf("%7.4lf  %7.4lf   loss : %8.20lf\n", *output, target[0], dnn.loss_);
			dnn.GradZero();
		    scanf("%*c"); //pause
		}
	}
	return 0;
}