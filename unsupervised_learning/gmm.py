import numpy as np
from scipy.stats import multivariate_normal
from utils.common_function import get_covariance_matrix, get_vector_norm
from utils.generate_data import generate_cluster_data
import matplotlib.pyplot as plt

class gaussian_mixture_model:
    def __init__(self, k, tolerance):
        self.k = k
        self.tolerance = tolerance
        self.responsibility_matrix_history = []
        
    def initiate_random_gaussian(self, X):
        N, D = X.shape
        self.pi = (1 / self.k) * np.ones(self.k) #This is also the prior
        self.mean = np.zeros((self.k, D))
        self.covariance = np.zeros((self.k, D, D))
        for k in range(self.k):
            self.mean[k] = X[np.random.choice(N),:]
            self.covariance[k] = get_covariance_matrix(X)
    
    def calculate_likelihood(self, X):
        N, D = X.shape
        multivariate_gaussian_likelihood = np.zeros((N,self.k))
        for k in range(self.k):
            mean = self.mean[k]
            cov = self.covariance[k]
            for n in range(N):
                multivariate_gaussian_likelihood[n, k] = multivariate_normal.pdf(X[n], mean, cov)
        return multivariate_gaussian_likelihood   #return array of size (N, k)
    
    def expectation_step(self, X):
        likelihood = self.calculate_likelihood(X)
        self.responsibility_matrix = (self.pi * likelihood) / (np.sum(likelihood, axis=1, keepdims=True)) #size (N, k)
        self.assignment = np.argmax(self.responsibility_matrix, axis=1) #size (N, )
        self.responsibility_matrix_history.append(self.responsibility_matrix.max(axis=1))
    
    def maximization_step(self, X):
        N = X.shape[0]
        for k in range(self.k):
            res = np.expand_dims(self.responsibility_matrix[:,k], axis=1)
            self.mean[k] = ((res * X).sum(axis=0)) / (res.sum())
            self.covariance[k] = (X - self.mean[k]).T.dot((X - self.mean[k]) * res) / res.sum()
            
        self.priors = self.responsibility_matrix.sum(axis=0) / N
        
    def check_converge(self):
        if len(self.responsibility_matrix_history) < 2:
            return False
        else:
            diff = get_vector_norm(self.responsibility_matrix_history[-1] - self.responsibility_matrix_history[-2], 2)
            return diff <= self.tolerance
        
if __name__ == '__main__':
    N, D, n_cluster = 600, 3, 3
    X, label = generate_cluster_data(N, D, n_cluster)
    k = 3
    tolerance = 1e-6
    model = gaussian_mixture_model(k, tolerance)
    model.initiate_random_gaussian(X)
    while not model.check_converge():
        model.expectation_step(X)
        model.maximization_step(X)
    x_axis = np.arange(N)
    plt.scatter(x_axis, X.mean(axis=1), c = model.assignment)
    plt.show()