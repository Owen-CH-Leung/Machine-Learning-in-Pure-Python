from utils.regularization import l1_regularization, l2_regularization, l1_l2_regularization
from utils.common_function import feed_forward, train_test_split, standardize
from utils.generate_data import get_cancer
import matplotlib.pyplot as plt
import numpy as np

class ridge_classifier:
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.regularizer = l1_regularization(alpha = 0.01)
    def fit(self, X, Y):        
        X = np.insert(X, 0, 1, axis=1)
        N, D = X.shape
        self.W = np.random.random(D) / np.sqrt(D)
        self.costs = []
        for i in range(self.n_iter):
            Y_pred = feed_forward(X, self.W, 0)
            cost = -1* (np.sum(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred)) + self.regularizer.cost(self.W))
            self.costs.append(cost)
            grad = -(Y - Y_pred).dot(X) + self.regularizer.gradient(self.W)
            self.W = self.W - self.lr * grad
        plt.plot(self.costs)
        plt.show()
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y_pred = feed_forward(X, self.W, 0)
        return Y_pred
    
class lasso_classifier:
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.regularizer = l2_regularization(alpha = 0.01)
    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)
        N, D = X.shape
        self.W = np.random.random(D) / np.sqrt(D)
        self.costs = []
        for i in range(self.n_iter):
            Y_pred = feed_forward(X, self.W, 0)
            cost = -1* (np.sum(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred)) + self.regularizer.cost(self.W))
            self.costs.append(cost)
            grad = -(Y - Y_pred).dot(X) + self.regularizer.gradient(self.W)
            self.W = self.W - self.lr * grad
        plt.plot(self.costs)
        plt.show()
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y_pred = feed_forward(X, self.W, 0)
        return Y_pred

class elastic_net:
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.regularizer = l1_l2_regularization(alpha = 0.01, l1_ratio=0.5)
    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)
        N, D = X.shape
        self.W = np.random.random(D) / np.sqrt(D)
        self.costs = []
        for i in range(self.n_iter):
            Y_pred = feed_forward(X, self.W, 0)
            cost = -1* (np.sum(Y * np.log(Y_pred) + (1-Y) * np.log(1-Y_pred)) + self.regularizer.cost(self.W))
            self.costs.append(cost)
            grad = -(Y - Y_pred).dot(X) + self.regularizer.gradient(self.W)
            self.W = self.W - self.lr * grad
        plt.plot(self.costs)
        plt.show()
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        Y_pred = feed_forward(X, self.W, 0)
        return Y_pred

if __name__ == '__main__':
    X, Y = get_cancer()
    X = standardize(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    n_iter , lr = 1000, 0.001
    ridge = ridge_classifier(n_iter, lr)
    ridge.fit(X_train, Y_train)
    ridge_pred = ridge.predict(X_test)
    lasso = lasso_classifier(n_iter, lr)
    lasso.fit(X_train, Y_train)
    lasso_pred = lasso.predict(X_test)
    elastic_net = elastic_net(n_iter, lr)
    elastic_net.fit(X_train, Y_train)
    elastic_pred = elastic_net.predict(X_test)
    print(f'Ridge Accuracy : {np.mean(Y_test == np.round(ridge_pred))}')
    print(f'Lasso Accuracy : {np.mean(Y_test == np.round(lasso_pred))}')
    print(f'Elastic-Net Accuracy : {np.mean(Y_test == np.round(elastic_pred))}')