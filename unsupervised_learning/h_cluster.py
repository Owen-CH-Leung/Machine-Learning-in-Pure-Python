import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from utils.generate_data import generate_cluster_data
import matplotlib.pyplot as plt

if __name__ == '__main__':
    N = 1000
    D = 3
    n_cluster = 3 
    link = 'single'
    metric = 'euclidean'
    X, true_label = generate_cluster_data(N, D, n_cluster)
    _ = dendrogram(linkage(X, method=link, metric=metric))
    
    #Determine n_cluster based on dendrogram
    model = AgglomerativeClustering(n_clusters = 3, linkage=link, affinity=metric)
    labels = model.fit_predict(X)
    
    pca = PCA(n_components = 2)
    X_transform = pca.fit_transform(X)
    
    plt.scatter(X_transform[:, 0], X_transform[:, 1], c = labels)
    plt.show()