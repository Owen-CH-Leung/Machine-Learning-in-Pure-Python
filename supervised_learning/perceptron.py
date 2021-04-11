import numpy as np
import matplotlib.pyplot as plt
from utils.generate_data import generate_classification_data
from utils.common_function import train_test_split

class Perceptron:
    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        D = X.shape[1]
        self.w = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        N = len(Y)
        self.costs = []
        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero(Y != Yhat)[0] #get index of the incorrect data 
            if len(incorrect) == 0:
                # we are done!
                break

            # choose a random incorrect sample
            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            # cost is incorrect rate
            c = len(incorrect) / float(N)
            self.costs.append(c)

    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == '__main__':
    N, D = 1000, 4
    learning_rate = 0.0001
    epochs = 5000
    X, Y = generate_classification_data(N, D, 2, 2, 1.5)
    Y[Y==0] = -1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    model = Perceptron()
    model.fit(X_train, Y_train, learning_rate = learning_rate, epochs=epochs)
    
    print("final w:", model.w, "final b:", model.b)
    print(f"score: {model.score(X_test, Y_test)}")
    plt.plot(model.costs)
    plt.show()