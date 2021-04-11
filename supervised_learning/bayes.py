import numpy as np
from utils.generate_data import generate_classification_data
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split

#Implement the Naive-Bayes : Assume independent features
class NaiveBayes:
    def fit(self, X, Y, smoothing=1e-2):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0),
                'var': current_x.var(axis=0) + smoothing,
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
        return np.argmax(P, axis=1)

#Implement the non-naive bayes , each feature is dependent on each other, cov != 0
class Bayes:
    def fit(self, X, Y, smoothing=1e-2):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean': current_x.mean(axis=0), #take mean column-wise, result is of dimension (D,)
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing, 
                #np.eye creates an identity matrix
                #np.cov creates an covariance matrix of dimension D*D (covariance between each column)
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in self.gaussians.items():
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1) #Return the position of max value row-wise
    
if __name__ == '__main__':
    X, Y = generate_classification_data(10000, 8, 4, 3, 1.85)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model_1 = NaiveBayes()
    model_1.fit(X_train, Y_train)
    model_1.predict(X_test)
    print(f"Naive_Bayes accuracy : {model_1.score(X_test, Y_test)}")
    model_2 = Bayes()
    model_2.fit(X_train, Y_train)
    model_2.predict(X_test)
    print(f"accuracy : {model_2.score(X_test, Y_test)}")
