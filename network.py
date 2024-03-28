import numpy as np
import random
from cProfile import Profile
from pstats import SortKey, Stats
from numpy import float128
from numpy.typing import NDArray

LEARNING_RATE = 0.01

class Neuron():
    act: float
    ws: NDArray[float128]
    bias: float128
    def __init__(self, activation: float, weights: NDArray[float128]):
        self.act = activation
        self.ws = weights 

    def calculate(self, previous_layer: 'Layer'):
        self.act = sum(n.act * self.ws[i] for i, n in enumerate(previous_layer.neurons))

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
            return layer.prev.neurons[w_idx].act * cost_deriv

        if layer.prev is None:
            raise Exception("should not use input layer weights")
        
        return layer.prev.neurons[w_idx].act *  sum([n.all_ws_foward(own_idx, layer.next) for n in layer.next.neurons]) * cost_deriv


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
            n.append(Neuron(0, weights[i]))
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
        self.layer_array = list(layers)
        self.input = layers[0]
        self.output = layers[-1]
        self.layer_count = len(layers)
        
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

    def back_prop(self, cost_deriv: float128):
        nd = []
        for l in self.layer_array:
            if l.prev is None:
                continue
            nd.append(np.zeros((l.size, l.prev.size)))

        layer_idx = self.layer_count - 1
        curr = self.output
        while curr.prev is not None:
            for i, n in enumerate(curr.neurons):
                for wi, w in enumerate(curr.neurons[i].ws):
                    # print(f"set w from {curr.neurons[i].ws[wi]} to {w - LEARNING_RATE * n.w_effect_on_cost(wi, i, curr, cost_deriv)}")
                    # curr.neurons[i].ws[wi] = w - LEARNING_RATE * n.w_effect_on_cost(wi, i, curr, cost_deriv)
                    nd[layer_idx][i][wi] = n.w_effect_on_cost(wi, i, curr, cost_deriv)
                    # curr.neurons[i].ws[wi] = random.random()
            curr = curr.prev
            layer_idx -= 1 
        return nd

    def train(self, epochs: int, train_set: list[tuple[NDArray[float128], NDArray[float128]]]):
        nd = []
        for l in self.layer_array:
            if l.prev is None:
                nd.append(None)
                continue
            nd.append(np.zeros((l.size, l.prev.size)))
            
        for ep in range(epochs):
            for ex in train_set:
                cost, d_cost = self.feed_example(ex[0], ex[1])
                print(f"{ep} cost: {cost}, out = {self.output.neurons[0].act}")
                nd += self.back_prop(d_cost)
                # self.print()
        for li, layer in enumerate(nd):
            for ni, n in enumerate(layer):
                for wi, w in enumerate(n):
                    self.layer_array[li].neurons[ni].ws[wi] -= LEARNING_RATE * w
        

    def print(self):
        curr = self.input
        while curr is not None:
            curr.print()
            curr = curr.next

def arr_from(*l: float) -> NDArray[float128]:
    a = NDArray(len(l))
    for i, e in enumerate(l):
        a[i] = e
    return a

train = [
    (arr_from(1, 1, 2), arr_from(3)),
    # (arr_from(2, 0, 1), arr_from(1)),
    (arr_from(2, 1, 5), arr_from(7)),
    # (arr_from(3, 1, 3), arr_from(6)),
    # (arr_from(4, 0, 2), arr_from(2)),
    # (arr_from(5, 1, 4), arr_from(9)),
    # (arr_from(6, 0, 2), arr_from(4)),
    # (arr_from(3, 1, 2), arr_from(5)),
    # (arr_from(7, 0, 3), arr_from(4)),
    # (arr_from(4, 1, 5), arr_from(9)),
    # (arr_from(8, 0, 1), arr_from(7)),
    # (arr_from(2, 1, 0), arr_from(2)),
    # (arr_from(6, 1, 1), arr_from(8)),
    # (arr_from(9, 0, 4), arr_from(5)),
    # (arr_from(3, 0, 1), arr_from(2)),
    # (arr_from(5, 1, 3), arr_from(8)),
    # (arr_from(7, 1, 2), arr_from(9)),
    # (arr_from(4, 1, 0), arr_from(4)),
    # (arr_from(5, 0, 2), arr_from(3)),
    # (arr_from(6, 1, 4), arr_from(11)),
    # (arr_from(8, 0, 3), arr_from(5)),
    # (arr_from(9, 1, 0), arr_from(9)),
    # (arr_from(3, 0, 2), arr_from(1)),
    # (arr_from(7, 1, 3), arr_from(10)),
    # (arr_from(1, 1, 1), arr_from(2)),
    # (arr_from(2, 0, 0), arr_from(2)),
    # (arr_from(4, 1, 2), arr_from(6)),
]


n = Network(
        Layer(3, np.zeros((3, 1), dtype=float128)),
        Layer(6, np.random.randn(6, 3).astype(float128)),
        Layer(1, np.random.randn(1, 6).astype(float128)),
        )


# with Profile() as profile:
#     print(f"out = {n.output.neurons[0].act}")
#     print(f"out_w = {n.output.neurons[0].ws}")
#     n.train(20, train)
#     n.feed_input(arr_from(1, 1, 2))
#     print(f"out = {n.output.neurons[0].act}")
#     print(f"out_w = {n.output.neurons[0].ws}")
#     (
#         Stats(profile)
#         .strip_dirs()
#         .sort_stats(SortKey.CALLS)
#         .print_stats()
#     )


n.train(40, train)

n.feed_input(arr_from(1, 1, 2))
print(f"out = {n.output.neurons[0].act}")
print(f"out_w = {n.output.neurons[0].ws}")

print()
print()
print()
print()

# n.feed_input(arr_from(10, 1, 2))
# print(f"out = {n.output.neurons[0].act}")
# print(f"out_w = {n.output.neurons[0].ws}")
