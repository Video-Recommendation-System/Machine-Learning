from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import math

# Fixing random state for reproducibility
np.random.seed(19680801)

X = load_iris()['data']
N = len(X)
#print(X)
#print(N)
#for sub_array in X:
#	print(sub_array)
#print(N)

X_embedded = TSNE(n_components=2).fit_transform(X)
k = int(round(math.sqrt(N)))
kmeans = KMeans(n_clusters=k, random_state=0).fit_transform(X_embedded)
kmeans_cluster = KMeans(n_clusters=k, random_state=0).fit(X_embedded)
cluster_labels = kmeans_cluster.labels_


for i in cluster_labels:
	i = i * 23

#colors = np.array(cluster_labels)

x = kmeans[:, 0]
y = kmeans[:, 1]


plt.scatter(x, y, s=50, c=cluster_labels, alpha=0.5)
plt.show()