import numpy as np

def sigmoid(Z):
    '''
    Parameters
    ----------
    Z : numpy array of size (N,)
        Apply the input to sigmoid function which has 
        bound = (0, 1)
        
    Returns
    -------
    numpy array of size (N,)
    '''
    return 1 / (1 + np.exp(-Z))

def feed_forward(X, W, B):
    '''
    Parameters
    ----------
    X : numpy array of size (N, D)
        The input data X
        
    W : numpy array of size (D, )
        The weights
    
    B : a scalar of float type
    
    Returns
    -------
    Matrix Multiplication of X and B, which will then be the input of 
    sigmoid function
    '''
    return sigmoid(X.dot(W) + B)

def cross_entropy_cost(T,Y_pred):
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

def cross_entropy_gradient_descent_w(W, learning_rate, T, Y_pred, X):
    '''
    Parameters
    ----------
    W : numpy array of size (D, )
        The weight vector 
        
    learning_rate : a single scalar of interger / float type
        control each steps W takes to descent along gradient.
        
    T : numpy array of size (N,)
        This is the target column Y
        
    Y_pred : numpy array of size (N,)
        The predicted probabilities of Y
        
    X : numpy array of size (N, D)
        The input data X
        
    Returns
    -------
    Numpy array of size (D,), corresponding to the weight vector

    '''
    return W - learning_rate * (X.T.dot(Y_pred - T))

def cross_entropy_gradient_descent_bias(B, learning_rate, T, Y_pred):
    '''
    Parameters
    ----------
    B : numpy array of size (D, )
        The bias term 
        
    learning_rate : a single scalar of interger / float type
        control each steps W takes to descent along gradient.
        
    T : numpy array of size (N,)
        This is the target column Y
        
    Y_pred : numpy array of size (N,)
        The predicted probabilities of Y
        
    Returns
    -------
    Numpy array of size (D,), corresponding to the weight vector

    '''
    return B - learning_rate * ((Y_pred - T).sum())

def get_R_square(Y, Y_hat):
    '''
    Parameters
    ----------
    Y : numpy array of size (N,)
        The real target Y
        
    Y_hat : numpy array of size (N,)
        The predicted outcome Y

    Returns
    -------
    a single scalar which represents the fitness of the model

    '''
    SS_res = np.sum((Y - Y_hat)**2)
    Y_bar = Y.mean()
    SS_tot = np.sum((Y - Y_bar)**2)
    return 1 - (SS_res /  SS_tot)

def one_hot_encoding(X):
    '''
    Parameters
    ----------
    X : categorical numpy array of size (N, )
        Input categorical column

    Returns
    -------
    target_X : numpy array of size (N, D)
        one-hot encoded numpy array of size (N, D), where D is the unique number of categories
    
    unique : list 
        Contains a list of unique value found in input X
    '''
    data_X = X
    unique = sorted(np.unique(data_X))
    D = len(unique)
    N = data_X.shape[0] 
    target_X = np.zeros((N, D))
    for n in range(D):
        x = (data_X == unique[n])
        target_X[x,n] = 1
    return target_X, unique

def shuffle_data(X, Y):
    '''        
    Returns
    -------
    Shuffled X, Y 
    '''
    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], Y[idx]

def train_test_split(X, Y, test_size=0.3, shuffle=True):
    '''
    Parameters
    ----------
    X : numpy array of size (N, D)
        The input data X
        
    Y : numpy array of size (N, )
        The target data Y
    
    test_size : float
        Determine the size of train / test dataset
    
    shuffe : boolean
        Determine whether to shuffle data
        
    Returns
    -------
    X_train, X_test, Y_train, Y_test    
    '''
    if shuffle == True:
        X, Y = shuffle_data(X, Y)
    N = Y.shape[0]
    split_idx = int(N * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    return X_train, X_test, Y_train, Y_test

def standardize(X, flag_1d = False):
    '''
    Parameters
    ----------
    X : numpy array of size (N, D)
        The input data X
    
    Returns
    -------
    standardized X
    '''

    m = X.mean(axis=0)
    std = X.std(axis=0)
    X_std = (X - m) / std
    return X_std

def min_max_standardize(X, target_max, target_min):
    '''
    Parameters
    ----------
    X : numpy array of size (N, D)
    target_max : target max value after transformation
    target_min : target min value after transformation

    Returns
    -------
    Transformed X with value range from (target_min, target_max)

    '''
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    X_scaled = X_std * (target_max - target_min) + target_min
    return X_scaled

def get_vector_norm(X, k):
    '''
    Parameters
    ----------
    X : numpy array of size (N, )
        
    k : integer
        controls the order of norm 

    Returns
    -------
    A float which represents the vector norm
    '''
    X_abs = np.abs(X)
    return np.sum(np.power(X_abs, k)) ** (1/k)
    #return np.linalg.norm(X, k)

def get_covariance_matrix(X):
    N = X.shape[0]
    return (1 / (N-1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
    #return np.cov(X.T, ddof=1)