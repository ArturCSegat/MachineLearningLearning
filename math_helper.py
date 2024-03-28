import numpy as np
from numpy.typing import NDArray
from numpy import float128

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):

    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def sum_arrays_in_list(list1: list[NDArray[float128]], list2: list[NDArray[float128]]):
    for i, _ in enumerate(list1):
        list1[i] += list2[i]

