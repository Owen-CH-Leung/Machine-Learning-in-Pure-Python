import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits, load_breast_cancer

def generate_one_dim_data(N):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record

    Returns
    -------
    X and Y which are numpy array of size (N, ) and size (N, )
    '''
    X = np.random.randn((N)) * 2.5 + 5  #generate random data from normal dist with mean = 5, var = 6.25
    b = np.random.randint(1,100) #generate one single random integer from 1 to 100
    a = np.random.random() #generate random float from 0 to 1
    Y = a * X + b + np.random.normal(loc=1, scale=1.2, size=N) #Add Noise
    return X, Y

def generate_continuous_linear_data(N, D, plot=True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    D : a single scalar of integer type
        Indicates the number of dimensions
    
    plot : a boolean flag
        Control whether to make a overview plot 
        
    Returns
    -------
    X and Y which are numpy array of size (N, D) and size (N, )
    '''
    #generate random data from normal dist with mean = 5, var = 6.25
    X = np.random.randn(N, D) * 2.5 + 5
    #Create gussian noise
    B = np.ones((N,1))
    X = np.concatenate((B, X), axis=1) #Add a bias term of ONE
    w = np.linspace(start=1, stop= N//D, num = D)
    Y = X.dot(w) + np.random.normal(loc=5, scale = 10, size=N) #create noise to Y
    if plot == True:
        x_plot = X.mean(axis=1)
        plt.scatter(x_plot, Y)
    return X, Y

def generate_continuous_linear_data_with_outlier(N, D, plot=True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    D : a single scalar of integer type
        Indicates the number of dimensions
    
    plot : a boolean flag
        Control whether to make a overview plot 
        
    Returns
    -------
    X and Y which are numpy array of size (N, D) and size (N, )
    '''
    dict_X = {}
    for i in range(D):
        dict_X[f'X{i}'] = np.random.randn(N) + (2 + i) # Generate linear data with mean = 2
    X = np.zeros((N,D))
    for i in range(D):
        X[:,i] = dict_X[f'X{i}'] 
    Y = 4 * X.mean(axis=1) + np.random.normal(loc=0, scale=0.5, size=N)
    last_N = int(0.1*N)
    for i in range(1, last_N):
        Y[-i] += 8 + np.random.randn()
    if plot == True:
        x_plot = X.mean(axis=1)
        plt.scatter(x_plot, Y)
    return X, Y

def generate_XOR_data(N = 1000, plot=True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    plot : a boolean flag
        Control whether to make a overview plot 
        
    Returns
    -------
    X and Y which are numpy array of size (N, 2) and size (N, )
    '''
    if N % 4 != 0:
        raise ValueError("N must be a muliple of 4")
    s = N//4
    X1 = np.random.uniform(low = 2, high=3, size=(s, 2)) #Generate data with x, y range = (2 - 3, 2 - 3)   
    X2 = np.random.uniform(low = 1, high=2, size=(s, 2)) #Generate data with x, y range = (1 - 2, 1 - 2)   
    X3 = np.random.uniform(low = 2, high=3, size=(s, 2)) - np.array([0, 1]) #Generate data with x, y range = (2 - 3, 1 - 2)
    X4 = np.random.uniform(low = 1, high=2, size=(s, 2)) + np.array([0, 1]) #Generate data with x, y range = (1 - 2, 2 - 3)
    X = np.concatenate((X1, X2, X3, X4), axis=0)
    Y = np.array([0] * s + [0] * s + [1] * s + [1] * s)
    if plot == True:
        plt.scatter(X[:,0], X[:,1], c= Y)
        plt.show()
    return X, Y
    
def generate_sparse_data(N,D, density, plot=True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    D : a single scalar of integer type
        Indicates the numnber of dimensions
        
    density : float
        Controls the level of density / sparseness
        
    plot : a boolean flag
        Control whether to make a plot 
        
    Returns
    -------
    X which are numpy array of size (N, D)

    '''
    sparse_X = sparse.random(N,D,density=density)
    X = sparse_X.toarray()
    true_w = np.zeros(D)
    for i in range(D//10):
        true_w[i] = np.random.random() #Only first D//10 columns matter
    Y = X.dot(true_w)
    if plot:
        plt.figure(figsize=(12, 12))
        plt.spy(sparse_X, markersize=1)
        plt.show()
    return X, Y, true_w

def generate_donut_data(N, plot = True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record

    plot : a boolean flag
        Control whether to make a plot 
        
    Returns
    -------
    X and Y which are numpy array of size (N, 2) and size (N, )
    '''
    R_inner = 8
    R_outer = 16

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))
    
    if plot:
        plt.scatter(X[:,0], X[:,1], c=Y)
        plt.show()
    return X, Y

def generate_classification_data(N, D, n_class, n_informative, class_similarity, balanced = True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    D : a single scalar of integer type
        Indicates the numnber of dimensions
        
    n_class : a single scalar of integer type
        Indicates number of distinct class
        
    class_similarity : a single scalar of float type
        Larger values spread out the clusters/classes and make the classification task easier
        
    balanced : boolean type
        a flag to control whether to create balanced or unbalanced class dataset

    Returns
    -------
    X and Y which are numpy array of size (N, D) and size (N, )

    '''
    if balanced == True:
        weights = np.array([1 / n_class] * n_class)
    elif balanced == False:
        weights = np.array([0.8] + [(0.2/n_class)] * (n_class-1))   
    
    X, Y = make_classification(n_samples=N, n_features=D, 
                               n_classes = n_class, n_informative = n_informative,
                               weights = weights, class_sep = class_similarity)
    return X, Y
        
def get_digits():
    '''        
    Returns
    -------
    X and Y which are numpy array of size (1797, 64) and size (1797, ) with 10 distinct classes
    '''
    X, Y = load_digits(return_X_y = True)
    return X, Y

def get_cancer():
    X, Y = load_breast_cancer(return_X_y = True)
    return X, Y
    
def generate_cluster_data(N, D, n_cluster, plot=True):
    '''
    Parameters
    ----------
    N : a single scalar of integer type
        Indicates the number of record
        
    D : a single scalar of integer type
        Indicates the numnber of dimensions
        
    n_cluster : a single scalar of integer type
        Indicates number of distinct cluster
        
    plot : a boolean flag
        Control whether to make a plot 

    Returns
    -------
    X : a numpy array of size (N, D)
    '''
    idx = np.linspace(start=0, stop=N, num=n_cluster+1).astype(np.int32)
    mean = np.arange(1, n_cluster+1)
    X = np.zeros((N, D))
    membership = np.zeros(N)
    p1 , p2 = 0, 1
    for i in range(n_cluster):
        if p2 == n_cluster:
            p2 = None
            start = idx[p1]   
            n = N - start 
            X[start:] = np.random.random((n, D)) + mean[i]
            membership[start:] = p1
        else:
            start, end = idx[p1], idx[p2]    
            n = end - start 
            X[start:end] = np.random.random((n, D)) + mean[i]
            membership[start:end] = p1
            p1 += 1
            p2 += 1
    if plot:
        x_axis = np.arange(N)
        plt.scatter(x_axis, X.mean(axis=1), c = membership)
        plt.show()
    member = np.unique(membership)
    return X, member