import numpy as np
N = 100
P = 4

W = np.array([[7.5, 0.5, 2, 4]]).T
def f(x: np.ndarray, w: np.ndarray): 
    return (x * w).sum() + np.random.normal(0, 100)

X = np.random.normal(0, 50, (N, P))
for x in X:
    x[0] = 1

Y = np.array([f(x, W) for x in X])

Wa = np.linalg.pinv(X.T @ X) @ X.T @ Y
Ya = np.array([f(x, Wa) for x in X])

TSS = ((Y - np.average(Y))**2).sum()
RSS = ((Ya- Y)**2).sum()

Var = ((Ya - Y) ** 2).sum() / N - P - 1

m = np.linalg.inv(X.T @ X)

for i in range(P):
    Se = np.sqrt(Var * m[i][i])
    print(f"SE-{i} = {Se} ---- Tstat = {Wa[i] / Se}")

print(f"R² = {1 - RSS / TSS}")
print(f"F = {((TSS-RSS)/P)/(RSS/(N-P-1))}")



