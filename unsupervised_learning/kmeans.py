from utils.generate_data import generate_cluster_data
from utils.common_function import get_vector_norm
import numpy as np
from copy import deepcopy

class kmeans:
    def __init__(self, k):
        self.k = k
        
    def initiate_centroid(self, X):
        N, D = X.shape
        self.membership = np.zeros(N)
        self.centroid = np.zeros((self.k, D))
        for k in range(self.k):
            idx = np.random.choice(N)
            self.centroid[k] = X[idx]
    
    def calculate_membership_hard(self, X):
        N, D = X.shape        
        self.cost = np.zeros(N)
        self.old_membership = deepcopy(self.membership)
        for n in range(N):
            closest_dist = float('inf')
            member = 0 
            for k in range(self.k):
                diff = X[n] - self.centroid[k]
                distance = get_vector_norm(diff, 2) #get euclidean distance
                if distance < closest_dist:
                    closest_dist = distance
                    member = k
            self.membership[n] = member
            self.cost[n] = closest_dist
        print(f'Total Cost : {self.cost.sum()}')
        
    def recalculate_centroid_hard(self, X):
        for k in range(self.k):
            new_mean = X[self.membership==k].mean(axis=0)
            self.centroid[k] = new_mean
    
    def convergence_hard(self):
        if np.all(self.old_membership == self.membership):
            return True
        else:
            return False
    def calculate_membership_soft(self, X, beta):
        N, D = X.shape        
        self.cost = 0
        self.exponent = np.zeros((N, self.k))
        for n in range(N):
            for k in range(self.k):
                diff = X[n] - self.centroid[k]
                distance = diff.dot(diff) #
                self.exponent[n,k] = np.exp(-beta * distance)
                
        R = self.exponent / self.exponent.sum(axis=1, keepdims=True)
        for k in range(self.k):
            diff = X - self.centroid[k]
            sq_dist = (diff * diff).sum(axis=1)
            self.cost += (R[:,k] * sq_dist).sum()
        print(f'Total Cost : {self.cost}')
        
    def recalculate_centroid_soft(self, X):
        R = self.exponent / self.exponent.sum(axis=1, keepdims=True)
        self.centroid = R.T.dot(X) / R.sum(axis=0, keepdims=True).T
            
    def get_membership_soft(self):
        self.membership = np.argmax(self.exponent, axis=1)
            
if __name__ == '__main__':
    N, D, n_cluster = 5000, 4, 5
    beta = 1
    n_iter = 50
    X, member = generate_cluster_data(N, D, n_cluster)
    model = kmeans(n_cluster)
    model.initiate_centroid(X)
    #model.calculate_membership_soft(X, beta)
    # model.calculate_membership_hard(X)
    # while not model.convergence_hard():
    #     model.recalculate_centroid_hard(X)
    #     model.calculate_membership_hard(X)
        
   
    for i in range(n_iter):
        model.calculate_membership_soft(X, beta)
        model.recalculate_centroid_soft(X)
    model.get_membership_soft()