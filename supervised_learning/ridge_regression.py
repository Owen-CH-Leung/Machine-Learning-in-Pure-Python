from utils.generate_data import generate_continuous_linear_data_with_outlier
from utils.common_function import get_R_square
import numpy as np
import matplotlib.pyplot as plt

class ridge_regression:
    def __init__(self, X,Y):           #Assuming 2D matrix
        self.X = X
        self.Y = Y
    def fit(self, lambda_):      #lambda_ is just a scalar 
        #Closed form solution for Ridge Regression
        X = self.X   
        Y = self.Y
        D = self.X.shape[1]      #The identity matrix should be of shape (D,D)
        self.W_ridge = np.linalg.solve(lambda_ * (np.eye(D)) + X.T.dot(X), X.T.dot(Y)) # Ridge Solution
        self.W_MLE = np.linalg.solve(X.T.dot(X), X.T.dot(Y)) #MLE solution
    def predict(self, X):
        Y_hat_ridge = X.dot(self.W_ridge)
        Y_hat_MLE = X.dot(self.W_MLE)
        return Y_hat_ridge, Y_hat_MLE
    
if __name__ == '__main__':
    N = 50
    D = 2
    lambda_ = 10
    X, Y = generate_continuous_linear_data_with_outlier(N, D, plot=True)
    model = ridge_regression(X, Y)
    model.fit(lambda_)
    Y_hat_ridge, Y_hat_MLE = model.predict(X)
    
    R_square_MLE = get_R_square(Y, Y_hat_MLE)
    R_square_ridge = get_R_square(Y, Y_hat_ridge)
    print(f"R_Square for MLE : {R_square_MLE}")
    print(f"R_Square for Ridge : {R_square_ridge}")
    
    plt.scatter(X.mean(axis=1), Y)
    plt.plot(sorted(X.mean(axis=1)), sorted(Y_hat_MLE), label='Maximum Likelihood')
    plt.plot(sorted(X.mean(axis=1)), sorted(Y_hat_ridge), label='map')
    plt.legend()
    plt.show()