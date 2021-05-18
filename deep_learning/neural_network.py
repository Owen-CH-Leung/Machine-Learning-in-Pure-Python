from utils.generate_data import get_iris
from utils.deeplearning_utils import ActivationFunction, LossFunction, ActivationFunctionDerivative
from utils.common_function import one_hot_encoding
import numpy as np
import matplotlib.pyplot as plt

def feed_forward(X, W1, W2, W3, B1, B2, B3, func = 'sigmoid'):
    if func == 'sigmoid':
        act_func = ActivationFunction.sigmoid
    elif func == 'tanh':
        act_func = ActivationFunction.tanh
    elif func == 'relu':
        act_func = ActivationFunction.relu
    Z1 = W1.T.dot(X) + B1
    A1 = act_func(Z1)
    Z2 = W2.T.dot(A1) + B2
    A2 = act_func(Z2)
    Z3 = W3.T.dot(A2) + B3
    Y_pred = ActivationFunction.softmax(Z3)
        
    params = {'A1': A1, 'A2': A2,
              'Z1': Z1, 'Z2': Z2, 'Z3': Z3}
    return Y_pred, params

def back_propagation(params, T, Y_pred, W3, W2, X, func = 'sigmoid'):
    if func == 'sigmoid':
        dri_func = ActivationFunctionDerivative.sigmoid
    elif func == 'tanh':
        dri_func = ActivationFunctionDerivative.tanh
    elif func == 'relu':
        dri_func = ActivationFunctionDerivative.relu
        
    dW3 = params['A2'].dot(T - Y_pred) #same shape as W3
    dB3 = np.sum(T - Y_pred, axis=0, keepdims=True).T #same shape as B3
    dW2 = params['A1'].dot((W3.dot((T - Y_pred).T) * dri_func(params['A2'])).T)
    dB2 = np.sum(W3.dot((T - Y_pred).T) * dri_func(params['A2']), axis=1, keepdims=True)
    dW1 = X.dot((W2.dot(W3.dot((T - Y_pred).T) * dri_func(params['A2'])) * dri_func(params['A1'])).T)
    dB1 = np.sum(W2.dot(W3.dot((T - Y_pred).T) * dri_func(params['A2'])) * dri_func(params['A1']), axis=1, keepdims=True)
    
    gradient_params = {'dW3': dW3, 'dB3': dB3, 
              'dW2': dW2, 'dB2': dB2, 
              'dW1': dW1, 'dB1': dB1}
    return gradient_params

def gradient_descent(W1, W2, W3, B1, B2, B3, gradient_params, lr=0.001):
    W1 = W1 + lr * gradient_params['dW1']
    W2 = W2 + lr * gradient_params['dW2']
    W3 = W3 + lr * gradient_params['dW3']
    B1 = B1 + lr * gradient_params['dB1']
    B2 = B2 + lr * gradient_params['dB2']
    B3 = B3 + lr * gradient_params['dB3']
    return W1, W2, W3, B1, B2, B3

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total
    
if __name__ == '__main__':
    X, Y = get_iris()
    T, _ = one_hot_encoding(Y)
    n_iter = 1000

    N, D = X.shape
    X = X.T
    M1 = 8
    M2 = 6
    K = 3
    for func in ['sigmoid', 'tanh', 'relu']:
        W1 = np.random.randn(D, M1) / np.sqrt(D)
        W2 = np.random.randn(M1, M2) / np.sqrt(M1)
        W3 = np.random.randn(M2, K) / np.sqrt(M2)
        B1 = np.expand_dims(np.random.randn(M1), 1)
        B2 = np.expand_dims(np.random.randn(M2), 1)
        B3 = np.expand_dims(np.random.randn(K), 1)
        loss_list = []
        
        for i in range(n_iter):
            Y_pred , params = feed_forward(X, W1, W2, W3, B1, B2, B3, func)
            grad_param = back_propagation(params, T, Y_pred.T, W3, W2, X)
            loss_list.append(LossFunction.CrossEntropy(T, Y_pred.T))
            W1, W2, W3, B1, B2, B3 = gradient_descent(W1, W2, W3, B1, B2, B3, grad_param)
        
        plt.plot(loss_list)
        plt.show()
        Y_pred_label = np.argmax(Y_pred.T, axis=1)
        print(f"Classification Rate using {func}: {classification_rate(Y, Y_pred_label)}")
    