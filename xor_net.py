#!/usr/bin/env python3

import numpy as np
from typing import List, NamedTuple, Tuple, Callable, Any, Dict
from functools import reduce
from collections import namedtuple

ActivationFunction = Callable[[np.array], np.array]
Weights = np.array
Bias = np.array
NeuronLayer = Tuple[Weights, Bias, ActivationFunction]
LayerResult = namedtuple("LayerResult", ["Z", "A"])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compose_functions(f: Callable, g: Callable) -> Callable:
    def fog(x: Any) -> Any:
        return g(f(x))

    return fog


def print_var(**varsd):
    return print(", ".join(f"{k}: {v}" for k, v in varsd.items()))


def create_neuron_layers(
    neurons_per_layer: List[int], activation_functions: List[ActivationFunction]
) -> List[NeuronLayer]:
    return [
        (np.random.randn(o, i), np.zeros((o, 1)), f)
        for i, o, f in zip(
            neurons_per_layer[:-1], neurons_per_layer[1:], activation_functions
        )
    ]


def forward_propagation(
    inputs: np.array, neuron_layer_list: List[NeuronLayer]
) -> List[LayerResult]:
    def partial_forward_propagation(weights, bias, activation_fn):
        def _forward_propagation(A_Z_list):
            Z = np.dot(weights, A_Z_list[-1].A) + bias
            return A_Z_list + [LayerResult(Z, np.squeeze(activation_fn(Z)))]

        return _forward_propagation

    partial_forward_propagation_list = [
        partial_forward_propagation(w, b, f) for w, b, f in neuron_layer_list
    ]
    recursive_forward_propagation = reduce(
        compose_functions, partial_forward_propagation_list
    )
    return recursive_forward_propagation([LayerResult(0, inputs)])


def cross_entropy_cost(
    An: np.array, outputs: np.array, number_of_training_examples: int
):
    return np.squeeze(
        -np.sum(
            np.multiply(outputs, np.log(An)) + np.multiply(1 - outputs, np.log(1 - An))
        )
        / number_of_training_examples
    )


def sqr_error_cost(An: np.array, outputs: np.array, number_of_training_examples: int):
    return np.sum(np.square(An - outputs)) / number_of_training_examples


def sqr_error_cost_derivative(An, outputs):
    return 2 * (An - outputs)


def backward_prop(
    inputs: np.array,
    outputs: np.array,
    A_Z_list: List[LayerResult],
    neuron_layer_list: List[NeuronLayer],
    number_of_training_examples: int,
    derivatives: Dict[str, Callable[[np.array], np.array]],
    cost_derivative_function: Callable[[np.array, np.array, int], float],
) -> List[Tuple[np.array, np.array, np.array]]:
    def _backward_prop(
        dZn: np.array,
        layer_weights_n: Weights,
        layer_result: np.array,
        Z: np.array,
        derivative_function: Callable[[np.array], np.array],
    ) -> Tuple[np.array, np.array, np.array]:
        print_var(a=layer_weights_n)
        dZl = np.multiply(
            np.dot(layer_weights_n, dZn), derivative_function(layer_result)
        )
        dWl = np.dot(layer_result, dZl.T) / number_of_training_examples
        dbl = np.sum(dZl, axis=1, keepdims=True) / number_of_training_examples
        return (dZl, dWl, dbl)

    dAn = cost_derivative_function(A_Z_list[-1].A, outputs, number_of_training_examples)
    dZn = derivatives[neuron_layer_list[-1][2].__name__](A_Z_list[-1].A) * dAn
    dWn = np.dot(dZn, A_Z_list[-2].A.T)
    dbn = np.sum(dZn, axis=0, keepdims=True)
    print_var(dAn=dAn, dZn=dZn, dWn=dWn, dbn=dbn)
    _A_Z_list = [0, inputs] + A_Z_list
    h = list(zip(neuron_layer_list[1:], _A_Z_list[::-1][2:]))
    partial_gradient_list = [
        _backward_prop(dZn, w, lr.A, lr.Z, derivatives[f.__name__])
        for (w, _, f), lr in h
    ]
    return partial_gradient_list + [(dZn, dWn, dbn)]


def update_parameters(neuron_layer_list: List[NeuronLayer], grads, learning_rate):
    return [
        (w - learning_rate * dW, b - learning_rate * db, f)
        for (w, b, f), (_, dW, db) in list(zip(neuron_layer_list, grads))
    ]


def model(
    inputs: np.array,
    outputs: np.array,
    neurons_per_layer: List[int],
    activation_functions: List[ActivationFunction],
    num_of_iters: int,
    learning_rate: float,
    cost_function: Callable[[np.array, np.array, int], float],
    number_of_trainig_examples: int,
    derivatives: Dict[str, Callable[[np.array], np.array]],
):
    neuron_layer_list = create_neuron_layers(neurons_per_layer, activation_functions)
    for i in range(0, num_of_iters + 1):
        layer_results = forward_propagation(inputs, neuron_layer_list)
        # print(f"lr: {layer_results}")
        cost = cost_function(layer_results[-1].A, outputs, number_of_trainig_examples)

        grads = backward_prop(
            inputs,
            outputs,
            layer_results,
            neuron_layer_list,
            number_of_trainig_examples,
            derivatives,
            sqr_error_cost,
        )

        neuron_layer_list = update_parameters(neuron_layer_list, grads, learning_rate)

        if i % 100 == 0:
            # print(f"An:{layer_results[-1]}, o:{outputs}")
            print("Cost after iteration# {:d}: {:f}".format(i, cost))

    return neuron_layer_list


def predict(inputs, trained_neuron_layer_list: List[NeuronLayer]):
    predict_neuron_layer_list = forward_propagation(inputs, trained_neuron_layer_list)
    yhat = predict_neuron_layer_list[-1][1]
    yhat = np.squeeze(yhat)
    return np.greater(yhat, 0.5)


np.random.seed(1)

# The 4 training examples by columns
inputs = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# The outputs of the XOR for every example in X
outputs = np.array([[0, 1, 1, 0]])

# No. of training examples
number_of_training_examples = inputs.shape[1]

# Test 2X1 vector to calculate the XOR of its elements.
# Try (0, 0), (0, 1), (1, 0), (1, 1)
X_test = np.array([[1, 0, 0, 1], [1, 0, 1, 0]])
neurons_per_layer = [2, 2, 1]
activation_functions = [np.tanh, np.tanh, sigmoid]
derivatives = {
    np.tanh.__name__: lambda x: 1 - np.power(x, 2),
    sigmoid.__name__: lambda x: np.multiply(1, (1 - x)),
}
num_of_iters = 100000
learning_rate = 0.3

trained_parameters = model(
    inputs,
    outputs,
    neurons_per_layer,
    activation_functions,
    num_of_iters,
    learning_rate,
    sqr_error_cost,
    number_of_training_examples,
    derivatives,
)

y_predict = predict(X_test, trained_parameters)

print(f"Neural Network prediction for example {X_test} is {y_predict}")
