# RianDNN
Deep neural networks library for C++ <br/>
I'm making this for use in my machine learning projects

## Features

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

## Example

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

#### Training

```cpp
for (int i = 0; i < 10000; i++) { //epoch
	double input[110];
	for (int j = 0; j < 100; j++) {
		input[j] = 0.5f;
	}
	double target[2] = { 0.3f, 0.7f};

	/*Forward*/
	double* output = dnn.Forward(input,target);

	/*Optimize*/
	if (i % 5 == 0) {
		dnn.Optimize();
	}
}
```


#### Evaluating

```cpp
double* output = dnn.Forward(input);
for (int i = 0; i < dnn.output_num_; i++)
	printf("%5.3lf\n", output[i]);
```
