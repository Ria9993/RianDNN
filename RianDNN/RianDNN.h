#pragma once

#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
using namespace std;

/*
enum class Type {
	Input = 0,
	Hidden,
	Output
};
enum class Activation {
	None = 0,
	ReLU
};
*/

namespace RianDNN {
	class Layer {
	public:
		int last_node_num_;
		int node_num_;
		string activation_;
		//None
		//ReLU
		vector<vector<double>> weight_;
		//[nowLayer_node][lastLayer_node]
		vector<double> bias_;
		vector<double> result_;
		vector<double> local_gradient_;
		~Layer() {
		}
	};

	class DNN {
	public:
		int input_num_;
		int layer_num_;
		vector<Layer> layer_;
		double learning_rate;
		DNN() {
			layer_num_ = 0;
		}
		~DNN() {
		}
		void AddLayer(int node_num, string activation);
		double* Forward(double* input);
		void Optimize(double loss);
		double GetAct(string activation, double x);
	};

	double DNN::GetAct(string activation, double x) {
		if (activation == "ReLU") {
			return std::fmax(0, x);
		}
		else { //"None"
			return x;
		}
		return 0;
	}
	void DNN::AddLayer(int node_num, string activation) {
		layer_num_++;
		layer_.resize(layer_num_);

		Layer* new_layer = &layer_[layer_num_ - 1];
		new_layer->node_num_ = node_num;
		new_layer->activation_ = activation;

		if (layer_num_ == 1)
			new_layer->last_node_num_ = input_num_;
		else
			new_layer->last_node_num_ = layer_[layer_num_ - 2].node_num_;
		new_layer->weight_.resize(new_layer->last_node_num_);
		//weight xavier initialization
		random_device rd;
		mt19937 rnd(rd());
		normal_distribution <double> xavier_init(0,(double)2 / (new_layer->last_node_num_ + new_layer->node_num_));
		new_layer->weight_.resize(new_layer->node_num_);
		for (int i = 0; i < new_layer->node_num_; i++) {
			new_layer->weight_[i].resize(new_layer->last_node_num_);
			for (int j = 0; j < new_layer->last_node_num_; j++)
				new_layer->weight_[i][j] = xavier_init(rnd);
		}
		new_layer->bias_.resize(new_layer->node_num_);
		for (int i = 0; i < new_layer->node_num_; i++)
			new_layer->bias_[i] = 0.0f;
		new_layer->result_.resize(new_layer->node_num_);
		new_layer->local_gradient_.resize(new_layer->node_num_);
		return;
	}
	double* DNN::Forward(double* input) {
		for (int n = 0; n < layer_num_; n++) {
			Layer* now = &layer_[n];
			for (int i = 0; i < now->node_num_; i++) {
				now->result_[i] = now->bias_[i];
				for (int j = 0; j < now->last_node_num_; j++) {
					if (n == 0) //input
						now->result_[i] += now->weight_[i][j] * input[j];
					else 
						now->result_[i] += now->weight_[i][j] * layer_[n-1].result_[j];
				}
				now->result_[i] = GetAct(now->activation_, now->result_[i]);
			}
		}
		return &layer_[layer_num_-1].result_[0];
	}
	void DNN::Optimize(double loss) {

	}
}