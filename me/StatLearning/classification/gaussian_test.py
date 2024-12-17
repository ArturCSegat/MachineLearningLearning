import matplotlib.pyplot as plt
import numpy as np

mu = 0 
se = 4

g = lambda x: 1 / np.sqrt(np.pi * 2 * (se**2)) * np.e ** (-1 * (x - mu)**2 / (2 * se**2))

X = np.arange(-10, 10)
Y = g(X)
plt.plot(X, Y)
plt.show()

