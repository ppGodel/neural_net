#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Callable
ActivationFunction = Callable[[np.array], np.array]
Weights = np.array
Bias = np.array
NeuronLayer = Tuple[Weights, Bias, ActivationFunction]
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def create_neuron_layers(neurons_per_layer: List[int],
                         activation_functions: List[ActivationFunction]) \
    -> List[NeuronLayer]:
    return [(np.random.randn(o,i),np.zeros((o, 1)), f)for i,o,f in zip(neurons_per_layer[:-1], neurons_per_layer[1:], activation_functions)]
    

def forward_prop(inputs: np.array, neuron_layer_list: List[NeuronLayer]) -> List[np.array]:
    p = inputs
    l= []
    for weights, bias, activation_function in neuron_layer_list:
        z = np.dot(weights, p) + bias
        p = activation_function(z)
        l.append(p)
    return l
    # return [activation_function(np.dot(weights, inputs) + bias) for weights, bias, activation_function in neuron_layer_list]

def calculate_cost(An, Y):
    cost = -np.sum(np.multiply(Y, np.log(An)) +  np.multiply(1-Y, np.log(1-An)))/m
    cost = np.squeeze(cost)

    return cost

def backward_prop(inputs: np.array, outputs: np.array,
                  layer_result_list: List[np.array],
                  neuron_layer_list: List[NeuronLayer],
                  m: int) -> List[Tuple[np.array, np.array, np.array]]:
    def _backward_prop(dZn: np.array, layer_weights_n: Weights,  layer_result: np.array) \
        -> Tuple[np.array, np.array, np.array]:
        dZl = np.multiply(np.dot(layer_weights_n.T, dZn), 1 - np.power(layer_result, 2))
        dWl = np.dot(dZl, layer_result.T) / m
        dbl = np.sum(dZl, axis=1, keepdims=True)/m
        return (dZl, dWl, dbl)
        
    dZn = layer_result_list[-1] - outputs
    dbn = np.sum(dZn, axis=1, keepdims=True)/m
    dWn = np.dot(dZn, layer_result_list[-2].T)
    _layer_result_list = [inputs] + layer_result_list
    h = list(zip(neuron_layer_list[1:],_layer_result_list[::-1][2:]))
    print(h)
    partial_gradient_list = [_backward_prop(dZn, w, A) for (w, _, _), A in h]
    return partial_gradient_list + [(dZn, dWn, dbn)]    
    

def update_parameters(neuron_layer_list: List[NeuronLayer], grads, learning_rate):
    return [(w - learning_rate*dW, b - learning_rate*db, f)
            for (w,b,f), (_, dW, db) in list(zip(neuron_layer_list, grads))]
        
def model(inputs: np.array, outputs: np.array, neurons_per_layer: List[int],
          activation_functions: List[ActivationFunction], num_of_iters: int,
          learning_rate: float, m: int):
    neuron_layer_list = create_neuron_layers(neurons_per_layer, activation_functions)
    for i in range(0, num_of_iters+1):
        layer_results = forward_prop(inputs, neuron_layer_list)

        cost = calculate_cost(layer_results[-1], outputs)

        grads = backward_prop(inputs, outputs, layer_results, neuron_layer_list, m)

        neuron_layer_list = update_parameters(neuron_layer_list, grads, learning_rate)

        if(i%100 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))

    return neuron_layer_list

def predict(inputs, trained_neuron_layer_list: List[NeuronLayer]):
    predict_neuron_layer_list = forward_prop(inputs, trained_neuron_layer_list)
    yhat = predict_neuron_layer_list[-1]
    yhat = np.squeeze(yhat)
    return np.greater(yhat, 0.5)

    
np.random.seed(1)

#The 4 training examples by columns
inputs = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

#The outputs of the XOR for every example in X
outputs = np.array([[0, 1, 1, 0]])

#No. of training examples
m = inputs.shape[1]

#Test 2X1 vector to calculate the XOR of its elements. 
#Try (0, 0), (0, 1), (1, 0), (1, 1)
X_test = np.array([[1], [1]])
neurons_per_layer = [2,2,1]
activation_functions = [np.tanh, np.tanh, sigmoid]
num_of_iters = 100000
learning_rate = 0.3

trained_parameters = model(inputs, outputs, neurons_per_layer, activation_functions,
                           num_of_iters, learning_rate, m)

y_predict = predict(X_test, trained_parameters)

print(f"Neural Network prediction for example {X_test} is {y_predict}")
