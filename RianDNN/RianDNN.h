#pragma once

#include <random>
#include <cmath>
#include <vector>
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
		DNN() {
			layer_num_ = 0;
		}
		~DNN() {
		}
		void AddLayer(int node_num, string activation);
		double* Forward(double* input);
	};

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
		for (int i = 0; i < new_layer->last_node_num_; i++) {
			new_layer->weight_[i].resize(new_layer->node_num_);
			for (int j = 0; j < new_layer->node_num_; j++)
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
		return NULL;
	}
}