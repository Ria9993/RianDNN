#pragma once

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

namespace RianDNN {
	class Layer {
	public:
		int last_node_num_;
		int node_num_;
		string activation_; //"None" or "ReLU"
		vector<vector<double>> weight_; //[nowLayer_node][lastLayer_node]
		vector<vector<double>> weight_grad_; //local gradient
		vector<double> grad_; //global gradient
		vector<double> bias_;
		vector<double> result_;
		vector<double> back_pass_;
		~Layer() {
		}
	};

	class DNN {
	public:
		//Hyperparam
		int input_num_;
		double learning_rate;

		int output_num_;
		int layer_num_;
		vector<Layer> layer_;

		int forward_step_; //for calc avg(local_decent)
		double loss_sum_;
		double loss_;
		DNN() {
			layer_num_ = 0;
			forward_step_ = 0;
			loss_sum_ = 0;
		}
		~DNN() {
		}
		void AddLayer(int node_num, string activation);
		double* Forward(double* input); //just forward
		double* Forward(int* input);
		double* Forward(double* input, double* target); //forward for optimize
		void Optimize(double* target);
		inline double GetAct(string activation, int layer_num, double x); //activation function
		inline double GetActDif(string activation, int layer_num, double x); //differential
		//inline double GetSoftmaxDif(); //Softmax must on outputlayer
		double GetLoss(double* target); //default = MSE(Mean Squared Error)
		double GetLossDif(double target, double x); //loss derivation
		void GradZero(); //reset gradient
	};

	inline double DNN::GetAct(string activation, int layer_num, double x) {
		if (activation == "ReLU") {
			return std::fmax(0, x);
		}
		else if (activation == "Softmax") { //Softmax is must with CEE loss
			double max = 0;
			for (int i = 0; i < layer_[layer_num].node_num_; i++) {
				if (i == 0)
					max = layer_[layer_num].result_[i];
				else if (max < layer_[layer_num].result_[i]) {
					max = layer_[layer_num].result_[i];
				}
			}
			double sum = 0;
			for (int i = 0; i < layer_[layer_num].node_num_; i++) {
				sum += expf(layer_[layer_num].result_[i] - max);
			}
			return expf(x - max) / sum;
		}
		else { //"None"
			return x;
		}
		return 0;
	}
	inline double DNN::GetActDif(string activation, int layer_num, double x) {
		if (activation == "ReLU") {
			return x > 0 ? 1 : 0;
		}
		else if (activation == "Softmax") {
			//x는 이미 계산된 Softmax(z)
			return (x * (1 - x));
		}
		else { //"None"
			return 1;
		}
		return 0;
	}
	double DNN::GetLoss(double* target) {
		double loss = 0;
		for (int i = 0; i < layer_[layer_num_ - 1].node_num_; i++) {
			loss += powf((target[i] - layer_[layer_num_ - 1].result_[i]), 2);
		}
		loss_ = loss / layer_[layer_num_ - 1].node_num_;
		loss_sum_ += loss_;
		return loss_sum_ / forward_step_;
	}
	double DNN::GetLossDif(double target, double x) {
		return -(target - x) / layer_[layer_num_ - 1].node_num_;
	}
	void DNN::AddLayer(int node_num, string activation) {
		layer_num_++;
		output_num_ = node_num;
		layer_.resize(layer_num_);

		Layer* new_layer = &layer_[layer_num_ - 1];
		new_layer->node_num_ = node_num;
		new_layer->activation_ = activation;

		if (layer_num_ == 1)
			new_layer->last_node_num_ = input_num_;
		else
			new_layer->last_node_num_ = layer_[layer_num_ - 2].node_num_;

		/* weight HE initialization (sqrt(2) * sqrt(2/(in+out))) */
		random_device rd;
		mt19937 rnd(rd());
		//normal_distribution <double> HE_init(0, sqrtf(2) * sqrtf((float)2 / (new_layer->last_node_num_ + new_layer->node_num_)));
		normal_distribution <double> HE_init(0, (float)2 / (new_layer->last_node_num_ + new_layer->node_num_));
		new_layer->weight_.resize(new_layer->node_num_);
		new_layer->weight_grad_.resize(new_layer->node_num_);
		for (int i = 0; i < new_layer->node_num_; i++) {
			new_layer->weight_[i].resize(new_layer->last_node_num_);
			new_layer->weight_grad_[i].resize(new_layer->last_node_num_);
			for (int j = 0; j < new_layer->last_node_num_; j++) {
				new_layer->weight_[i][j] = HE_init(rnd);
				new_layer->weight_grad_[i][j] = 0.0f;
			}
		}
		new_layer->grad_.resize(new_layer->node_num_);
		new_layer->bias_.resize(new_layer->node_num_);
		new_layer->result_.resize(new_layer->node_num_);
		new_layer->back_pass_.resize(new_layer->node_num_);
		for (int i = 0; i < new_layer->node_num_; i++) {
			new_layer->bias_[i] = 0.1f;
			new_layer->result_[i] = 0.0f;
			new_layer->grad_[i] = 0;
			new_layer->back_pass_[i] = 0;
		}
		return;
	}
	double* DNN::Forward(double* input) {
		for (int n = 0; n < layer_num_; n++) {
			Layer* now = &layer_[n];
			if (n == 0) { //First Hidden layer
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * input[j];
						now->weight_grad_[i][j] = input[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
				}
			}
			else {
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * layer_[n - 1].result_[j];
						now->weight_grad_[i][j] = layer_[n - 1].result_[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
				}
			}
		}
		return &layer_[layer_num_ - 1].result_[0];
	}
	double* DNN::Forward(int* input) {
		for (int n = 0; n < layer_num_; n++) {
			Layer* now = &layer_[n];
			if (n == 0) { //First Hidden layer
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * (double)input[j];
						now->weight_grad_[i][j] = (double)input[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
				}
			}
			else {
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * layer_[n - 1].result_[j];
						now->weight_grad_[i][j] = layer_[n - 1].result_[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
				}
			}
		}
		return &layer_[layer_num_ - 1].result_[0];
	}
	double* DNN::Forward(double* input, double* target) {
		for (int n = 0; n < layer_num_; n++) {
			Layer* now = &layer_[n];
			if (n == 0) { //First Hidden layer
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * input[j];
						now->weight_grad_[i][j] = input[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
					now->grad_[i] += GetActDif(now->activation_, n, now->result_[i]);
				}
			}
			else {
				for (int i = 0; i < now->node_num_; i++) {
					now->result_[i] = now->bias_[i];
					for (int j = 0; j < now->last_node_num_; j++) {
						now->result_[i] += now->weight_[i][j] * layer_[n - 1].result_[j];
						now->weight_grad_[i][j] = layer_[n - 1].result_[j];
					}
					now->result_[i] = GetAct(now->activation_, n, now->result_[i]);
					now->grad_[i] += GetActDif(now->activation_, n, now->result_[i]);
				}
			}
		}
		Layer* output = &layer_[layer_num_ - 1];
		for (int i = 0; i < output->node_num_; i++) {
			output->back_pass_[i] += GetLossDif(target[i], output->result_[i]);
		}
		forward_step_++;
		GetLoss(target);
		return &layer_[layer_num_ - 1].result_[0];
	}
	void DNN::Optimize(double* target) {
		/*Get Avg derivation*/
		for (int n = 0; n < layer_num_; n++) {
			Layer* now = &layer_[n];
			for (int i = 0; i < now->node_num_; i++) {
				now->grad_[i] /= forward_step_;
				if (n == layer_num_ - 1) {
					now->back_pass_[i] /= forward_step_;
				}
				for (int j = 0; j < now->last_node_num_; j++) {
					now->weight_grad_[i][j] /= forward_step_;
				}
			}
		}
		/*BackPropagate*/
		for (int n = layer_num_ - 1; n >= 0; n--) {
			Layer* now = &layer_[n];
			for (int i = 0; i < now->node_num_; i++) {
				now->grad_[i] *= now->back_pass_[i];
			}
			for (int i = 0; i < now->node_num_; i++) {
				for (int j = 0; j < now->last_node_num_; j++) {
					now->weight_grad_[i][j] *= now->grad_[i];
					now->weight_[i][j] = now->weight_[i][j] - learning_rate * now->weight_grad_[i][j];
					if (n != 0) {
						layer_[n - 1].back_pass_[i] += now->weight_grad_[i][j];
					}
				}
				now->bias_[i] = now->bias_[i] - learning_rate * now->grad_[i];
			}
		}
		GradZero();
	}
	void DNN::GradZero() {
		forward_step_ = 0;
		loss_sum_ = 0;
		for (int n = layer_num_ - 1; n >= 0; n--) {
			Layer* now = &layer_[n];
			for (int i = 0; i < now->node_num_; i++) {
				now->grad_[i] = 0;
				now->back_pass_[i] = 0;
				for (int j = 0; j < now->last_node_num_; j++) {
					now->weight_grad_[i][j] = 0;
				}
			}
		}
		return;
	}
}