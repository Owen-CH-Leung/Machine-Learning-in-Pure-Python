import numpy as np
from utils.generate_data import generate_sparse_data

class l1_regularization:
    """ Regularization for Lasso """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def cost(self, w):
        return self.alpha * np.linalg.norm(w, 1)

    def gradient(self, w):
        return self.alpha * np.sign(w)

class l2_regularization:
    """ Regularization for Ridge """
    def __init__(self, alpha):
        self.alpha = alpha
    
    def cost(self, w):
        return self.alpha * w.T.dot(w)

    def gradient(self, w):
        return self.alpha * w

class l1_l2_regularization:
    """ Regularization for Elastic Net """
    def __init__(self, alpha, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def cost(self, w):
        l1_contr = self.l1_ratio * np.linalg.norm(w, 1)
        l2_contr = (1 - self.l1_ratio) * w.T.dot(w) 
        return self.alpha * (l1_contr + l2_contr)

    def gradient(self, w):
        l1_contr = self.l1_ratio * np.sign(w)
        l2_contr = (1 - self.l1_ratio) * w
        return self.alpha * (l1_contr + l2_contr) 