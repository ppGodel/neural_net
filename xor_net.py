#!/usr/bin/env python3

from functools import partial
import numpy as np
from typing import List, NamedTuple, Tuple, Callable

ActivationFunction = Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]
Weights = np.ndarray
Bias = np.ndarray
NeuronLayer = Tuple[Weights, Bias, ActivationFunction]
Model = NamedTuple('Model', [("train", Callable[[List[List[int]], List[int], int],List[NeuronLayer]])])


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivate(z):
    sigmoid_value = sigmoid(z)
    return sigmoid_value * (1 - sigmoid_value)

def tanh_derivate(z):
    tanh_value = np.tanh(z)
    return 1 - tanh_value**2

def relu_derivative(z: np.ndarray):
    return np.where(z < 0, 0, 1)

def sigmoid_derivate_from_activation(activation_value: np.ndarray):
    return activation_value * (1 - activation_value)

def tanh_derivate_from_activation(activation_value: np.ndarray):
    return 1 - (activation_value**2)

def relu_derivative_from_activation(activation_value: np.ndarray):
    return np.where(activation_value > 0, 1, 0)

def linear(z):
    return z

def linear_derivative(activation_value):
    return np.ones_like(activation_value)

def create_neuron_layers(neurons_per_layer: List[int],
                         activation_functions: List[ActivationFunction]) \
    -> List[NeuronLayer]:
    return [(np.random.randn(next_layer_size,actual_layer_size), # weights
             np.zeros((next_layer_size, 1)), # bias
             activation_function) #activation_function
            for actual_layer_size,next_layer_size,activation_function \
            in zip(neurons_per_layer[:-1], # from first until one before last (-1)
                   neurons_per_layer[1:], # from second to last
                   activation_functions)]
    
#inputs : a matrix were  each "column" is a case, each row is a "node" from the current layer (x)
def forward_prop(inputs: np.ndarray, neuron_layer_list: List[NeuronLayer]) -> List[np.ndarray]:
    activation_value = inputs
    layer_activation_values= []
    for weights, bias, activation_function in neuron_layer_list: # ineherenly secuential
        z:np.ndarray = np.dot(weights, activation_value) + bias
        activation_value :np.ndarray = activation_function[0](z)
        layer_activation_values.append(activation_value)
    return layer_activation_values
    # return [activation_function(np.dot(weights, inputs) + bias) for weights, bias, activation_function in neuron_layer_list]

def square_error(predicted:np.ndarray, desired: np.ndarray):
    return (desired - predicted)**2

def square_error_derivate(predicted:np.ndarray, desired: np.ndarray):
    return (desired - predicted)*2
def backward_prop(inputs: np.ndarray, outputs: np.ndarray,
                  layer_results_activation_list: List[np.ndarray],
                  neuron_layer_list: List[NeuronLayer]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

    m = inputs.shape[1]  # number of examples

    def _backward_prop_hidden(dA_next: np.ndarray, next_layer_weights: Weights,
                              current_activation: np.ndarray, activation_derivative: Callable) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # dA_next comes from the layer ahead
        # Compute dZ for current layer
        dZ = np.multiply(np.dot(next_layer_weights.T, dA_next), activation_derivative(current_activation))
        # Get activation from previous layer for dW calculation
        return dZ

    # Output layer gradients
    last_layer_activation_function = neuron_layer_list[-1][2]
    An = layer_results_activation_list[-1]

    # For square error: dL/dAn = 2(An - Y), then dL/dZn = dL/dAn * g'(An)
    dAn = 2 * (An - outputs)  # Cost derivative w.r.t. activation
    dZn = np.multiply(dAn, last_layer_activation_function[1](An))  # Apply activation derivative

    A_prev = layer_results_activation_list[-2] if len(layer_results_activation_list) > 1 else inputs
    dWn = np.dot(dZn, A_prev.T) / m
    dbn = np.sum(dZn, axis=1, keepdims=True) / m

    # Build list of (next_layer_weights, current_activation, current_activation_derivative)
    # We need to go backwards through hidden layers
    all_activations = [inputs] + layer_results_activation_list[:-1]  # Activations before each hidden layer

    # Pair: (layer_n+1_weights, layer_n_activation, layer_n_activation_derivative)
    # Start from second-to-last layer going backwards
    from functools import reduce

    def accumulate_grads(acc, idx):
        grads_list, current_dZ = acc

        # Get current layer info (going backwards from second-to-last)
        layer_idx = len(neuron_layer_list) - 2 - idx
        next_layer_weights = neuron_layer_list[layer_idx + 1][0]
        current_activation = layer_results_activation_list[layer_idx]
        current_activation_derivative = neuron_layer_list[layer_idx][2][1]
        prev_activation = all_activations[layer_idx]

        # Compute gradients for current layer
        dZ = _backward_prop_hidden(current_dZ, next_layer_weights,
                                   current_activation, current_activation_derivative)
        dW = np.dot(dZ, prev_activation.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        return ([(dZ, dW, db)] + grads_list, dZ)

    # Process all hidden layers
    num_hidden_layers = len(neuron_layer_list) - 1
    if num_hidden_layers > 0:
        initial_acc = ([(dZn, dWn, dbn)], dZn)
        final_grads, _ = reduce(accumulate_grads, range(num_hidden_layers), initial_acc)
        return final_grads
    else:
        return [(dZn, dWn, dbn)]
    

def update_parameters(neuron_layer_list: List[NeuronLayer], grads:List[Tuple[np.ndarray, np.ndarray, np.ndarray]], learning_rate:float):
    return [(w - learning_rate*dW, b - learning_rate*db, f)
            for (w,b,f), (_, dW, db) in list(zip(neuron_layer_list, grads))]
        
def train(start_neuron_layer_list: List[NeuronLayer], learning_rate: float, inputs: List[List[int]],
          outputs: List[int], epochs: int= 1000)->List[NeuronLayer]:
    neuron_layer_list = start_neuron_layer_list
    for epoch in range(1, epochs+1):
        input_value = np.array(inputs).T
        output_value = np.array(outputs).reshape(1, -1)
        layer_results = forward_prop(input_value, neuron_layer_list)
        cost = square_error(layer_results[-1], output_value)
        grads = backward_prop(input_value, output_value, layer_results, neuron_layer_list)
        # print(grads)
        neuron_layer_list = update_parameters(neuron_layer_list, grads, learning_rate)

        if(epoch%100 == 0):
            print(f"Cost after iteration# {epoch}: {np.mean(cost)}")
            # print(f"net: {grads}")

    return neuron_layer_list

def create_neural_net_model(neurons_per_layer: List[int],
                 activation_functions: List[ActivationFunction],
                 learning_rate: float)-> Model:
    neuron_layer_list = create_neuron_layers(neurons_per_layer, activation_functions)
    print(f"initial rand model: {neuron_layer_list}")
    return Model(partial(train, neuron_layer_list, learning_rate))

def predict_classification(inputs, trained_neuron_layer_list: List[NeuronLayer]):
    predict_neuron_layer_list = forward_prop(inputs, trained_neuron_layer_list)
    yhat = predict_neuron_layer_list[-1]
    yhat = np.squeeze(yhat)
    return np.greater(yhat, 0.5)

def predict_regression(inputs, trained_neuron_layer_list: List[NeuronLayer]):
    predict_neuron_layer_list = forward_prop(inputs, trained_neuron_layer_list)
    yhat = predict_neuron_layer_list[-1]
    return np.squeeze(yhat)
    
np.random.seed(2)

#The 4 training examples by columns
inputs = [[0, 0], [0,1], [1,0], [1, 1]]

#The outputs of the XOR for every example in X
outputs = [0, 1, 1, 0]


#Test 2X1 vector to calculate the XOR of its elements. 
#Try (0, 0), (0, 1), (1, 0), (1, 1)
X_test = np.array([[1,1,0,0], [1,0,1,0]])
neurons_per_layer = [2,2,2,1]
activation_functions = [(np.tanh, tanh_derivate_from_activation),
                        (np.tanh, tanh_derivate_from_activation),
                        (sigmoid, sigmoid_derivate_from_activation)]
num_of_iters = 100000
learning_rate = 0.3

nn_model = create_neural_net_model(neurons_per_layer, activation_functions, learning_rate)

trained_model = nn_model.train(inputs, outputs, num_of_iters)
y_predict = predict_classification(X_test, trained_model)

print(f"Neural Network prediction for example {X_test} is {y_predict}")
print(f"{forward_prop(X_test, trained_model)}")



# Simple linear regression example
np.random.seed(42)

# Generate some linear data: y = 2x + 3
X_train = [[1], [2], [3], [4], [5]]
y_train = [5, 7, 9, 11, 13]

neurons_per_layer = [1, 1]  # 1 input, 1 output
activation_functions = [(linear, linear_derivative)]

nn_model = create_neural_net_model(neurons_per_layer, activation_functions, learning_rate=0.01)
trained_model = nn_model.train(X_train, y_train, epochs=10000)

# Test
X_test = np.array([[6, 7, 8]])
predictions = predict_regression(X_test, trained_model)
print(f"Neural Network prediction for example {X_test} is {y_predict}")
print(f"{forward_prop(X_test, trained_model)}")
print(f"Predictions: {predictions}")  # Should be close to [15, 17, 19]
