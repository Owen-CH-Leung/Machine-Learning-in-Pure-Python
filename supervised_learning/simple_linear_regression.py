from utils.common_function import get_R_square
from utils.generate_data import generate_one_dim_data
from sklearn.model_selection import train_test_split
import numpy as np

class simple_linear_regression():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]
    def fit(self):
        #Closed form solution for y = ax + b
        self.a_numerator = self.N * np.sum(self.X * self.Y) - (np.sum(self.X) * np.sum(self.Y)) #np.sum(X*Y) can be written as X.dot(Y)
        self.a_denominator = self.N * np.sum(self.X ** 2) - (np.sum(self.X)) ** 2
        self.b_numerator = (np.sum(self.X) * np.sum(self.X * self.Y)) - np.sum(self.Y) * np.sum(self.X**2)
        self.b_denominator = (np.sum(self.X))**2 - (self.N * np.sum(self.X ** 2)) #np.sum(X**2) can be written as X.dot(X)
        self.a = self.a_numerator / self.a_denominator
        self.b = self.b_numerator / self.b_denominator
    def predict(self, X):
        Y_hat = self.a * X + self.b
        return Y_hat

if __name__ == '__main__':
    N = 10000
    X, Y = generate_one_dim_data(N)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = simple_linear_regression(X_train, Y_train)
    model.fit()
    Y_hat = model.predict(X_test)
    R_square = get_R_square(Y_test, Y_hat)
    print(f"R_Square : {R_square}")