import numpy as np 

class ActivationFunction:
    @staticmethod
    def sigmoid(A):
        '''
        Parameters
        ----------
        A : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return 1 / (1 + np.exp(-1 * A))
    
    @staticmethod
    def tanh(A):
        '''
        Parameters
        ----------
        A : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return (np.exp(A) - np.exp(-A)) / (np.exp(A) + np.exp(-A))
    
    @staticmethod
    def relu(A):
        '''
        Parameters
        ----------
        A : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return np.maximum(A, 0)
    
    def softmax(A):
        '''
        Parameters
        ----------
        A : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        exp_X = np.exp(A)
        return exp_X / exp_X.sum(axis=0, keepdims=True)
    
class LossFunction:
    @staticmethod
    def BinaryCrossEntropy(T, Y_pred):        
        '''
        Parameters
        ----------
        T : numpy array of size (N,)
            The target column Y
            
        Y_pred : numpy array of size (N,)
            The predicted probabilities of Y
    
        Returns
        -------
        A single scalar cost which stands for the total error 
        '''
        return -1* np.sum(T * np.log(Y_pred) + (1-T) * np.log(1-Y_pred))
    
    @staticmethod
    def CrossEntropy(T, Y_pred):
        '''
        Parameters
        ----------
        T : numpy array of size (N, K) (K number of distinct class)
            The target column Y
            
        Y_pred : numpy array of size (N, K)
            The predicted probabilities of Y
    
        Returns
        -------
        A single scalar cost which stands for the total error 
        '''
        return -1 * (np.sum(T * np.log(Y_pred)))

class ActivationFunctionDerivative:
    @staticmethod
    def sigmoid(Z):
        '''
        Parameters
        ----------
        Z : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return ActivationFunction.sigmoid(Z) * (1 - ActivationFunction.sigmoid(Z))
    
    def tanh(Z):
        '''
        Parameters
        ----------
        Z : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return 1 - ((ActivationFunction.tanh(Z)) ** 2)
    
    def relu(Z):
        '''
        Parameters
        ----------
        Z : numpy array of size (M, N), derived by :
            W : numpy array of size (D, M)
            X : numpy array of size (D, N)
            
        Returns
        -------
        numpy array of size (M, N)
        '''
        return np.where(Z <= 0, 0, 1)