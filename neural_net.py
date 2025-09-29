#!/usr/bin/env python3

import numpy as np
from typing import List, Callable, Any, Dict, TypeVar, Union, Tuple
from functools import reduce


T = TypeVar("T")


def to_binary(number: int) -> List[bool]:
    return [bool(int(binary_value)) for binary_value in bin(number)[2:]]

def to_binary_fixed(number: int, depth) -> List[bool]:
    binary = to_binary(number)
    return [binary[position] if position < len(binary) else False for position in range(depth)]

def to_decimal(boolean_list: List[bool]) -> int:
    l_list =[
            int(value) * 2 ** posicion
            for posicion, value in enumerate(boolean_list[::-1])
        ] if len(boolean_list) > 0 else [0]
    return reduce(
        lambda n1, n2: n1 + n2,
        l_list)


def get_line_list_from_file(file_path: str) -> List[str]:
    return open(file_path, "r").read().splitlines()


def create_matrix_from_lines(lines: List[str], split_str: str) -> np.matrix:
    return np.matrix([line.split(split_str) for line in lines])


def read_matrix_from_file(file_path: str, split_str: str) -> np.matrix:
    return create_matrix_from_lines(get_line_list_from_file(file_path), split_str)


def create_matrix_from_str(str_values: str, split_str: str) -> np.matrix:
    return create_matrix_from_lines(str_values.split("\n"), split_str)


def matrix_map(map_dict: Dict[str, float], cell_type: str) -> float:
    return map_dict[cell_type]


def matrix_map_partial(map_dict: Dict[str, float]) -> Callable[[str], float]:
    def f(cell_type: str) -> float:
        return map_dict[cell_type]
    return f

def compose_functions(f: Callable, g: Callable) -> Callable:
    def fog(x:Any) -> Any:
        return g(f(x))
    return fog
def sigmoid(x: Union[int,float]) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: Union[int,float])-> float:
        return x * (1.0 - x)

def bool_to_float(x: bool)->float:
    return float(x)
     
def grather_than_zero(x: float)->bool:
    return x >= 0

def crear_digito_con_modelo(ultimo_digito: int, ciclos: int,tamanio: int, modelos_numericos: np.matrix)->List[Tuple[int, np.matrix]]:
    def _digito_a_modelo(digito:int):
        return  (np.random.uniform(0, 1, tamanio) < modelos_numericos[digito]).astype(int)
    return [(digito, _digito_a_modelo(digito)) for digito in np.random.randint(0, ultimo_digito, ciclos)]
# def nnural_net():
#     def train(train_input: List[int], gradient: Callable[[Any], float], activation: Callable[[Any], float]):
#         def setup_train(case:int, gradient: Callable[[Any], float], activation: Callable[[Any], float]):
#             def _train(weights: np.matrix) -> np.matrix:
#                 pass
#             return _train
#         return reduce(compose_functions, (setup_train(case_input, gradient, activation) for case_input in train_input))
#     return train

vmap = np.vectorize(matrix_map_partial({"n": 0.995, "g": 0.92, "b": 0.002}))
modelos_numericos = vmap(read_matrix_from_file("number_model.csv", " "))
numero_de_filas = 5
numero_de_columnas = 3
tamanio = numero_de_filas * numero_de_columnas
ciclos_de_entrenamiento = 15_000
ultimo_digito = 9
digitos = [x for x in range(ultimo_digito)]
profundidad_binario = len(to_binary(ultimo_digito))

weights = np.matrix(
    np.split(
        np.array(np.random.uniform(size=tamanio * profundidad_binario)),
        profundidad_binario,
    )
)


entradas_entrenamiento = crear_digito_con_modelo(ultimo_digito, ciclos_de_entrenamiento, tamanio, modelos_numericos)
entradas_prueba = np.random.randint(0, ultimo_digito, 5000)
valor_aceptacion = 0.9
output_function = sigmoid
fix_function = lambda x: bool(x >= 0.1)
gradient_function = sigmoid_derivative

for salida, modelo_entrada in entradas_entrenamiento:
    valores_binarios_esperados = to_binary_fixed(salida, profundidad_binario)
    entrada = np.full((profundidad_binario, tamanio), modelo_entrada)
    #new_weights = np.dot(weights, entrada.T)
    nw = []
    for valor_esperado_profundidad, weight in zip(valores_binarios_esperados, weights):
        sum_pesos = float(np.dot(weight, modelo_entrada.T))
        output = output_function(sum_pesos) #  activation fn
        error = valor_esperado_profundidad - output
        ajuste = gradient_function(output) *  error 
        nw.append(weight + (ajuste * modelo_entrada))
        

    weights = np.array(nw)

#test
print(f"{weights}")
res = []
for valor_prueba in entradas_prueba:
    modelo_entrada_entrenamiento = (np.random.uniform(0, 1, tamanio) < modelos_numericos[valor_prueba]).astype(int)
    valor_esperado_prueba = to_binary_fixed(valor_prueba, profundidad_binario)
    output_list = []
    for valor_esperado_profundidad, weight in zip(valor_esperado_prueba, weights):
        sum_pesos = float(np.dot(weight, modelo_entrada_entrenamiento.T))
        output = fix_function(output_function(sum_pesos))
        output_list.append(output)
    res.append((output_list, to_decimal(output_list), valor_prueba, to_decimal(output_list) == valor_prueba))
# print(f"{res}\n")
percentaje = np.mean([int(ac)for _,_,_, ac in res])
#print(res)
print(percentaje)
