'''
import numpy as np
from sklearn.manifold import TSNE
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
X_embedded.shape
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Fixing random state for reproducibility
np.random.seed(19680801)


N = len([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
X_embedded = TSNE(n_components=2).fit_transform(X)
x = X_embedded
y = X_embedded
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()