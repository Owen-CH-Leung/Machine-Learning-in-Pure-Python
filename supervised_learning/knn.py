import numpy as np
from sortedcontainers import SortedList
from utils.generate_data import generate_classification_data
from sklearn.model_selection import train_test_split

class Knn:
    def __init__(self,k):
        self.k = k
    
    def fit(self,X,Y):
        self.X = X
        self.Y = Y
    
    #Below implement uniform weighted knn, without vectorization
    def predict_uniform_weights(self, X):
        N = X.shape[0]
        self.Y_pred = np.zeros(N) #Get test dataset shape
        for i, x in enumerate(X): #Test dataset
            sl = SortedList() # stores (distance, class) tuples
            for j, xt in enumerate(self.X): #Train dataset
                diff = x - xt
                distance = diff.dot(diff) #Calculate the Manhatten Distance
                if len(sl) < self.k: #For the first 3 iteration
                    sl.add((distance, self.Y[j]))
                elif distance < sl[-1][0]:  #Check if the distance is smaller than the largest item in the list
                    del sl[-1]
                    sl.add((distance, self.Y[j]))
            #Next we get each neighbors class
            votes = {}
            for _ , c in sl:
                votes[c] = votes.get(c,0) + 1
                
            max_vote = 0
            max_vote_class = -1
            for k, v in votes.items():
                if v > max_vote:
                    max_vote = v
                    max_vote_class = k
            
            self.Y_pred[i] = max_vote_class
            
        return self.Y_pred

    def weighted_predict(self, X):
        N = X.shape[0]
        self.Y_pred = np.zeros(N) 
        for i, x in enumerate(X): 
            diff = x - self.X  
            distance_vector = np.diag(np.inner(diff, diff))  #like dot product with axis=1
            k_closest_index = np.argpartition(distance_vector, self.k)[:self.k] #get k smallest distance
            k_closest_dist = distance_vector[k_closest_index]
            corresponding_y = self.Y[k_closest_index]
            weights = k_closest_dist / np.sum(k_closest_dist)
            
            #Compare each unique Y and add its weights
            unique = set(corresponding_y)
            max_weights = 0
            max_vote = -1
            for y in unique:
                idx = np.where(corresponding_y == y)
                total_weights = np.sum(weights[idx])
                if total_weights > max_weights:
                    max_vote = y
            self.Y_pred[i] = max_vote
        return self.Y_pred
    
    def accuracy(self, Y):
        return np.mean(self.Y_pred==Y)

if __name__ == '__main__':
    X, Y = generate_classification_data(1000, 5, 3, 3, 0.9)
    model = Knn(5)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    model.fit(X_train, Y_train)
    model.predict_uniform_weights(X_test)
    print(f"accuracy : {model.accuracy(Y_test)}")
    model.weighted_predict(X_test)
    print(f"accuracy : {model.accuracy(Y_test)}")