from utils.common_function import get_R_square
from utils.generate_data import generate_continuous_linear_data
from sklearn.model_selection import train_test_split
import numpy as np

class multi_linear_regression():
    def __init__(self, X, Y):
        self.X = X   # X.shape = (N,D)
        self.Y = Y   # Y.shape = (N,)
    def fit(self):
        #Closed form solution for Y = aX + b, where X is a matrix instead of vector
        self.W = np.linalg.solve(self.X.T.dot(self.X), self.X.T.dot(self.Y)) # W.shape(D,)
    def predict(self, X):
        Y_hat = X.dot(self.W)
        return Y_hat
    
if __name__ == '__main__':
    N = 10000
    D = 5
    X, Y = generate_continuous_linear_data(N, D)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = multi_linear_regression(X_train, Y_train)
    model.fit()
    Y_hat = model.predict(X_test)
    R_square = get_R_square(Y_test, Y_hat)
    print(f"R_Square : {R_square}")