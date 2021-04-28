import numpy as np
from utils.generate_data import get_iris
from utils.common_function import standardize
import matplotlib.pyplot as plt

class LinearSVM:
    def __init__(self, C=1.0):
        self.C = C
    
    def initialize(self, X):
        N, D = X.shape
        self.N = N
        self.w = np.random.randn(D)
        self.b = 0
        
    def cost_function(self, pred): 
        #pred is of size (N,)
        #return a singl scalar
        return 0.5 * self.w.dot(self.w) + self.C * np.maximum(0, 1 - pred).sum()

    def gradient_descent(self, X, Y, lr=1e-5, n_iters=400):         
        # gradient descent
        losses = []
        for _ in range(n_iters):
          margins = Y * self._decision_function(X)
          loss = self.cost_function(margins)
          losses.append(loss)
          
          idx = np.where(margins < 1)[0]
          grad_w = self.w - self.C * Y[idx].dot(X[idx])
          self.w -= lr * grad_w
          grad_b = -self.C * Y[idx].sum()
          self.b -= lr * grad_b
          
        self.support_vector = np.where((Y * self._decision_function(X)) <= 1)[0]
        print("num SVs:", len(self.support_vector))
          
        print("w:", self.w)
        print("b:", self.b)
          
        # hist of margins
        m = Y * self._decision_function(X)
        plt.hist(m, bins=20)
        plt.show()
          
        plt.plot(losses)
        plt.title("loss per iteration")
        plt.show()

    def _decision_function(self, X):
        return X.dot(self.w) + self.b
    
    def predict(self, X):
        return np.sign(self._decision_function(X))
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

if __name__ == '__main__':
    X, Y = get_iris()
    mask = [(Y == 0) | (Y == 1)]
    X, Y = X[mask], Y[mask]
    Y[Y==0] = -1
    X = standardize(X)
    model = LinearSVM()
    model.initialize(X)
    model.gradient_descent(X, Y)
    print(f"Score : {model.score(X, Y)}")
    
    