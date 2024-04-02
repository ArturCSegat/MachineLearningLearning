import numpy as np
from numpy.typing import NDArray
from numpy import float128
import random

def arr_from(*l: float) -> NDArray[float128]:
    a = NDArray(len(l))
    for i, e in enumerate(l):
        a[i] = e
    return a


def training_data(ammount: int) -> list[tuple[NDArray[float128], NDArray[float128]]]:
    data = []
    for _ in range(ammount):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        data.append((arr_from(a, 1, b), arr_from(a+b)))
    return data
#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(z: float128):
    """ReLU function."""
    return np.maximum(0, z)

def relu_prime(z) -> float128:
    """Derivative of the ReLU function."""
    return float128(np.where(z > 0, 1, 0))

def sum_arrays_in_list(list1: list[NDArray[float128]], list2: list[NDArray[float128]]):
    for i, _ in enumerate(list1):
        list1[i] += list2[i]

