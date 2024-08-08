import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.argv[1] is None:
    print("please provide a polynomial degree")
    print("usage: python3 main.py N")
    exit()

N = int(sys.argv[1])
MAX_WEIGHT = 3
MIN_WEIGHT = -MAX_WEIGHT
LEN = (N+1)
BATCH = 100
ERROR = (BATCH/MAX_WEIGHT) * ((N-1.8) * N)

def predict(x: float, W) -> float:
    sum = 0
    for i in range(len(W)):
        sum += (x ** i) * W[i]
    
    return sum

def predict_weights(x, y, degree):
    length = degree + 1

    m = np.array([
            [sum(x ** (j+i)) for j in range(length)] for i in range(length)
        ] )

    A = np.linalg.inv(m)

    y_sum_col = np.array([sum(y * (x**i)) for i in range(length)]).reshape(length, 1)

    return A.dot(y_sum_col)

def MSE(y, y_) -> float:
    sum = 0
    l = len(y)
    for i in range(l):
        sum += (y[i] - y_[i]) ** 2
    return sum / l

def main():
    W = np.random.randint(MIN_WEIGHT, MAX_WEIGHT, (LEN)) + np.random.rand((LEN))
    print(f"real weights: {W}")

    X = np.random.uniform(-BATCH/10, BATCH/10, (BATCH))
    X_test = np.random.uniform(-BATCH/10, BATCH/10, (BATCH))
    Y = np.array([predict(x, W) for x in X]) + np.random.normal(0, ERROR, (BATCH))
    Y_test = np.array([predict(x, W) for x in X_test])

    mse_list = []
    n = 1
    W_: np.ndarray
    while (True):
        W_ = predict_weights(X, Y, n)
        Y_ = np.array([predict(x, W_) for x in X_test])
        mse = MSE(Y_test, Y_)
        mse_list.append(mse)
        print(mse)

        if len(mse_list) > 5:
            m = min(mse_list)
            i = mse_list.index(m)

            if i == 0:
                print(f"probable polynomial degree is {i+1} and its mse is {m}")
                break;
            if i == len(mse_list) - 1:
                n += 1
                continue 

            if mse_list[i-1] > m and mse_list[i+1] > m:
                print(f"probable polynomial degree is {i+1} and its mse is {m}")
                break;


        n += 1

    plt.scatter(X, Y)
    X.sort()
    plt.plot(X, [predict(x, W_) for x in X], color="red")
    plt.show()

    

if __name__ == "__main__":
    main()
