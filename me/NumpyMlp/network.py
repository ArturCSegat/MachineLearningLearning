import numpy as np
import random
from cProfile import Profile
from pstats import SortKey, Stats
from numpy import float128
from numpy.typing import NDArray
from math_helper import relu, relu_prime, sum_arrays_in_list
import math_helper

LEARNING_RATE = 0.001

class Neuron():
    act: float128
    ws: NDArray[float128]
    z_sum: float128
    def __init__(self, activation: float128, weights: NDArray[float128]):
        self.act = activation
        self.z_sum = relu_prime(activation)
        self.ws = weights 

    def calculate(self, previous_layer: 'Layer'):
         self.z_sum = float128(sum(n.act * self.ws[i] for i, n in enumerate(previous_layer.neurons)))
         self.act = relu(self.z_sum)

    def all_ws_foward(self, own_idx: int, own_layer: 'Layer') -> float:
        if own_layer.next is None:
            return 1

        wsum = 0
        for i, n in enumerate(own_layer.next.neurons):
            wsum += n.ws[own_idx] * n.all_ws_foward(i, own_layer.next)
        return wsum

    def w_effect_on_cost(self, w_idx: int, own_idx: int, layer: 'Layer', cost_deriv: float128):
        if layer.next is None:
            if layer.prev is None:
                raise Exception("Must have different input and output layers")
            return relu_prime(layer.prev.neurons[w_idx].z_sum) * cost_deriv

        if layer.prev is None:
            raise Exception("should not use input layer weights")
        
        return relu_prime(layer.prev.neurons[w_idx].z_sum) * sum([n.all_ws_foward(own_idx, layer.next) for n in layer.next.neurons]) * cost_deriv


class Layer():
    neurons: list[Neuron]
    next: 'Layer | None'
    prev: 'Layer | None'
    size: int

    def __init__(self, size: int, weights: NDArray[float128]):
        self.size = size
        self.next = None
        self.prev = None
        n: list[Neuron] = []
        for i in range (size):
            n.append(Neuron(float128(0), weights[i]))
        self.neurons = n
    def calculate(self, prev_layer: 'Layer'):
        for n in self.neurons:
            n.calculate(prev_layer)
    def print(self):
        out = ""
        for n in self.neurons:
            out += f" {n.act} "
        print(out)

class Network():
    input: Layer
    output: Layer
    layer_array: list[Layer]
    layer_count: int
    def __init__(self, *layers: Layer):
        self.input = layers[0]
        self.output = layers[-1]
        self.layer_count = len(layers)
        self.layer_array = list(layers)
        
        for i in range(len(layers) - 1):
            layers[i].next = layers[i + 1]
            layers[i + 1].prev = layers[i]

    def feed_foward(self):
        curr = self.input.next
        if curr is None:
            return
        while curr is not None:
            if curr.prev is None:
                return
            curr.calculate(curr.prev)
            curr = curr.next

    def feed_input(self, inp: NDArray[float128]):
        for i, _ in enumerate(self.input.neurons):
            self.input.neurons[i].act = inp[i]
        self.feed_foward()


    def feed_example(self,example: NDArray[float128], answer: NDArray[float128]) -> tuple[float128, float128]:
        self.feed_input(example)
        costs = float128(0)
        derived_costs = float128(0)
        for i, n in enumerate(self.output.neurons):
            costs += (n.act - answer[i]) ** 2
            derived_costs += (n.act - answer[i]) * 2
        
        return (costs / len(self.output.neurons), derived_costs / len(self.output.neurons))

    def back_prop(self, cost_deriv: float128) -> list[NDArray[float128]]:
        curr = self.output

        ls: list[NDArray[float128]] = []
        while curr.prev is not None:
            ws_ns = np.zeros((curr.size, curr.prev.size), dtype=float128)

            for i, n in enumerate(curr.neurons):
                der_sigmoid = n.act * (1 - n.act)
                for wi, _ in enumerate(curr.neurons[i].ws):
                    # print(f"set w from {curr.neurons[i].ws[wi]} to {w - LEARNING_RATE * n.w_effect_on_cost(wi, i, curr, cost_deriv)}")
                   ws_ns[i][wi] = n.w_effect_on_cost(wi, i, curr, cost_deriv) 
                    # curr.neurons[i].ws[wi] = random.random()
            ls.append(ws_ns)
            curr = curr.prev
        return ls

    def train(self, epochs: int, train_set: list[tuple[NDArray[float128], NDArray[float128]]]):
        for ep in range(epochs):
            ws: list[NDArray[float128]] = []
            for curr in self.layer_array.__reversed__():
                if curr.prev is None:
                    continue
                ws.append(np.zeros((curr.size, curr.prev.size), dtype=float128))


            for ex in train_set:
                cost, d_cost = self.feed_example(ex[0], ex[1])
                # print(f"{ep} cost: {cost}, out = {self.output.neurons[0].act}")
                sum_arrays_in_list(ws, self.back_prop(d_cost))
                # self.print()

            for i, _ in enumerate(ws):
                ws[i] /= len(train_set)

            for li, l in enumerate(self.layer_array.__reversed__()):
                if l.prev is None:
                    continue
                for ni, n in enumerate(l.neurons):
                    for wi, w in enumerate(n.ws):
                        self.layer_array[self.layer_count-li-1].neurons[ni].ws[wi] = w - LEARNING_RATE * ws[li][ni][wi]

    

    def print(self):
        curr = self.input
        while curr is not None:
            curr.print()
            curr = curr.next

n = Network(
        Layer(3, np.zeros((3, 1), dtype=float128)),
        Layer(10, np.random.randn(10, 3).astype(float128)),
        Layer(1, np.random.randn(1, 10).astype(float128)),
        )

n.train(150, math_helper.training_data(75))

n.feed_input(math_helper.arr_from(1, 1, 2))
print(f"out = {n.output.neurons[0].act}")
print(f"out_w = {n.output.neurons[0].ws}")

print()
print()
print()
print()

n.feed_input(math_helper.arr_from(2, 1, 5))
print(f"out = {n.output.neurons[0].act}")
print(f"out_w = {n.output.neurons[0].ws}")
