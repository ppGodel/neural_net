#!/usr/bin/env python3

import numpy as np
from typing import List, Callable, Any, Dict, TypeVar, Union
from functools import reduce
import numpy.random as rand
from numpy.lib.function_base import vectorize

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


vmap = np.vectorize(matrix_map_partial({"n": 0.995, "g": 0.92, "b": 0.002}))
modelos_numericos = vmap(read_matrix_from_file("number_model.csv", " "))
numero_de_filas = 5
numero_de_columnas = 3
tamanio = numero_de_filas * numero_de_columnas
ciclos_de_entrenamiento = 10_000
tasa_original = 0.15
tranqui = 0.999999998
ultimo_digito = 9
digitos = [x for x in range(ultimo_digito)]
profundidad_binario = len(to_binary(ultimo_digito + 1))

weights = np.matrix(
    np.split(
        np.array(rand.uniform(size=tamanio * profundidad_binario)),
        profundidad_binario,
    )
)

tasa = tasa_original
entradas_entrenamiento = rand.randint(0, ultimo_digito, ciclos_de_entrenamiento)
print(weights)

def sigmoid(x: Union[int,float])->float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: Union[int,float])-> float:
        return x * (1.0 - x)
    
def neural_net():
    def train():
        pass
    return trained_model


for train in range(ciclos_de_entrenamiento):
    numero_prueba = entradas_entrenamiento[train]
    modelo_entrada_entrenamiento = rand.uniform(0, 1, tamanio) < modelos_numericos[numero_prueba]
    valores_binarios_esperados = to_binary_fixed(numero_prueba, profundidad_binario)
    entrada = np.full((profundidad_binario, tamanio), modelo_entrada_entrenamiento)
    new_weights = np.dot(weights, entrada.T)
    
    #nw = []
    #for valor_esperado_profundidad, weight in zip(valores_binarios_esperados, weights):
    #    sum_pesos = float(np.dot(weight, modelo_entrada_entrenamiento.T))
    #    output = bool(0 <= sum_pesos) # activation fn
    #    if output != valor_esperado_profundidad:
    #        ajuste = tasa * (valor_esperado_profundidad - output)
    #        tasa = tranqui * tasa
    #        nw.append(weight + (ajuste * modelo_entrada_entrenamiento))
    #    else:
    #        nw.append(weight)
        

    #weights = np.array(nw)

#test
entradas_prueba = rand.randint(0, ultimo_digito, 1000)
print(f"{weights}")
print("t")
res = []
for valor_prueba in entradas_prueba:
    modelo_entrada_entrenamiento = rand.uniform(0, 1, tamanio) < modelos_numericos[valor_prueba]
    valor_esperado_prueba = to_binary_fixed(valor_prueba, profundidad_binario)
    output = []
    for valor_esperado_profundidad, weight in zip(valor_esperado_prueba, weights):
        sum_pesos = float(np.dot(weight, modelo_entrada_entrenamiento.T))
        output.append(0>=sum_pesos)
    res.append((to_decimal(output), valor_prueba, to_decimal(output) == valor_prueba))
# print(f"{res}\n")
percentaje = np.mean([int(ac)for _,_, ac in res])
print(tasa)
print(percentaje)
