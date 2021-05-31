# RianDNN
Deep neural networks library for C++ <br/>
I'm making this for use in my machine learning projects

# Features

#### Optimizer
```python
SGD(Stochastic Gradient Descent)
```
#### Activation Function
```python
ReLU,
Softmax, 
None
```
#### Loss Function
```python
MSE(Mean Squared Error)
```
#### Weight Initialize
```python
He Normal(Default)
```
#### Layer
```python
Dense
```
## To be update
```python
CNN,
LSTM
```

# Example

#### Include

```cpp
#include "RianDNN/RianDNN.h"
using namespace RianDNN;
```

#### Create Model

```cpp
RianDNN::DNN dnn;
dnn.learning_rate = 0.01f;
dnn.input_num_ = 10; //input Layer
dnn.AddLayer(30, "ReLU");
dnn.AddLayer(30, "ReLU");
dnn.AddLayer(30, "ReLU");
dnn.AddLayer(2, "Softmax");
```
#### Forward and Optimize
```cpp
//Just forward
dnn.Forward(input);
//Forward for optimize(Stacking Gradient)
dnn.Forward(input,target);
dnn.Forward(input2,target2);
//Update
dnn.Optimize();
```
#### Evaluating

```cpp
double* output = dnn.Forward(input);
for (int i = 0; i < dnn.output_num_; i++)
	printf("%7.5lf\n", output[i]);
```

#### Training (DQN Example)
```cpp
/*DQN model*/
RianDNN::DNN model;
model.learning_rate = 0.01f;
model.input_num_ = 10 * 10;
model.AddLayer(50, "ReLU");
model.AddLayer(50, "ReLU");
model.AddLayer(50, "ReLU");
model.AddLayer(4, "None");

/*Target Network*/
RianDNN::DNN model_target = model;
```
```cpp
for (int step = 0;; step++) {
	//Get state
	double input = game.StateToInt();

	//Forward
	double* output = model.Forward(input);
	
	//Action
	double reward = game.SendAction(max_idx(output));
	
	//Save Experience (s,a,r,s',done_mask)
	Memory.push_back(state, max_idx(output), reward, game.StateToInt(), game.End());
	
	/*Optimize*/
	if (step % 10 == 0) {
	
		/*Mini-Batch*/
		for(int i = 0; i < 20; i++) {
		
			int rnd_idx = Memory.rand_get();
			
			//get Q(s)
			double* tmp_output = model.Forward(Memory[rnd_idx].state);
			
			//get Q(s+1,a+1)
			double* tmp_next = model_target.Forward(Memory[end_idx].next_state);
			double tmp_next_Q = max(tmp_next);
			
			/*Target Update*/
			double Q_target[4];
			copy(tmp_output, tmp_output + 4, Q_target);
			Q_target[Memory[rnd_idx].action_] +=
				learning_rate *
					(Memory[rnd_idx].reward_ + 
					gamma * tmp_next_Q -
					Q_target[Memory[rnd_idx].action_]);
					
			//Forward for Optimize(Gradient Stacking)
			model.Forward(Memory[rnd_idx].state, Q_target);
		}
		
		//Update
		dnn.Optimize();
	}
	
}

//Target Netward Update
model_target = model;
```
# DQN Example(Repository)
[Ria9993/DQN-Ballgame](https://github.com/Ria9993/DQN-Ballgame)
```
