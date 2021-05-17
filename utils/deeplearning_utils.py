import numpy as np 

class ActivationFunction:
    @staticmethod
    def sigmoid(W, X):
        '''
        Parameters
        ----------
        W : numpy array of size (D, M)
        X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return 1 / (1 + np.exp(-1 * (W.T.dot(X))))
    
    @staticmethod
    def tanh(W, X):
        '''
        Parameters
        ----------
        W : numpy array of size (D, M)
        X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        x = W.T.dot(X)
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def relu(W, X):
        '''
        Parameters
        ----------
        W : numpy array of size (D, M)
        X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        x = W.T.dot(X)
        return np.maximum(x, 0)
    
    def softmax(W, X):
        '''
        Parameters
        ----------
        W : numpy array of size (M, K)
        X : numpy array of size (M, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        x = W.T.dot(X)
        exp_X = np.exp(x)
        return exp_X / exp_X.sum(axis=1, keepdims=True)
    

    
if __name__ == '__main__':
    N = 10
    D = 2
    M = 4
    K = 3
    X = np.random.random((D, N)) - 0.5
    W1 = np.random.random((D, M)) 
    W2 = np.random.random((M, K))
    z1 = ActivationFunction.sigmoid(W1, X)
    pred_y = ActivationFunction.softmax(W2, z1)