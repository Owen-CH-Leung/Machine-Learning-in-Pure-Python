from utils.common_function import sigmoid, feed_forward, cross_entropy_cost, cross_entropy_gradient_descent_w, cross_entropy_gradient_descent_bias
from utils.generate_data import generate_XOR_data
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    N = 1000
    X, Y = generate_XOR_data(N)
    XY = (X[:,0] * X[:,1]).reshape(N, 1)
    X = np.concatenate((XY, X), axis=1)
    D = 3
    learning_rate = 0.001
    cost = []
    W = np.random.randn(D) / np.sqrt(D)
    B = 0
    iterations = 50000
    for i in range(iterations):
        Y_pred = feed_forward(X, W, B)
        i_cost = cross_entropy_cost(Y, Y_pred)
        cost.append(i_cost)
        W = cross_entropy_gradient_descent_w(W, learning_rate, Y, Y_pred, X)
        B = cross_entropy_gradient_descent_bias(B, learning_rate, Y, Y_pred)
    accuracy = np.mean(Y == np.round(Y_pred))
    print(f"accuracy : {accuracy}")    
    plt.plot(cost)
    plt.show()