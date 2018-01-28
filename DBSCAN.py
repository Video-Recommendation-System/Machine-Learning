from __future__ import division, print_function
import itertools
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import math
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#PLACE X VECTOR HERE
#SET CSV FILE HERE
df = pd.read_csv('csv2.csv')
#PROVIDE THE CSV HEADER
X = df.as_matrix(["sentiment","/Science/Mathematics","/Science/Physics","/News/Business News","/Home & Garden/Home Appliances","/Science/Earth Sciences","/Reference","/Arts & Entertainment/Music & Audio/Radio","/Science","/Finance/Investing/Currencies & Foreign Exchange","/People & Society/Subcultures & Niche Interests","/Sports/Team Sports","/Reference/Humanities/History","/Health/Public Health","/Law & Government/Public Safety/Law Enforcement","/Sports/Team Sports/Baseball","/Games/Computer & Video Games","/Arts & Entertainment/Fun & Trivia/Flash-Based Entertainment","/Games/Computer & Video Games/Shooter Games","/Home & Garden/Kitchen & Dining","/Computers & Electronics/Software","/Arts & Entertainment/Music & Audio/World Music","/Games","/Sensitive Subjects","/Arts & Entertainment/TV & Video","/Reference/General Reference/Calculators & Reference Tools","/Business & Industrial/Energy & Utilities","/Computers & Electronics/Consumer Electronics","/Games/Roleplaying Games","/Arts & Entertainment","/Arts & Entertainment/Music & Audio/Dance & Electronic Music","/Arts & Entertainment/Music & Audio/Religious Music","/Arts & Entertainment/TV & Video/TV Shows & Programs","/Sports/Team Sports/Cricket","/Jobs & Education/Education/Teaching & Classroom Resources","/Arts & Entertainment/Music & Audio","/Law & Government/Public Safety","/Science/Engineering & Technology","/Food & Drink","/Arts & Entertainment/Performing Arts","/Home & Garden/Kitchen & Dining/Small Kitchen Appliances","/Arts & Entertainment/TV & Video/Online Video","/Business & Industrial","/Law & Government/Public Safety/Crime & Justice","/Arts & Entertainment/Music & Audio/Urban & Hip-Hop","/Computers & Electronics/Software/Multimedia Software","/Arts & Entertainment/Online Media","/Hobbies & Leisure","/Arts & Entertainment/Music & Audio/Music Equipment & Technology","/Arts & Entertainment/Music & Audio/Soundtracks","/Arts & Entertainment/Music & Audio/Rock Music","/People & Society/Religion & Belief","/Computers & Electronics/Software/Business & Productivity Software","/Books & Literature","/Computers & Electronics/CAD & CAM","/Games/Board Games/Chess & Abstract Strategy Games","/Internet & Telecom","/Health/Health Conditions/Infectious Diseases","/Online Communities","/Arts & Entertainment/Comics & Animation/Comics","/Business & Industrial/Energy & Utilities/Renewable & Alternative Energy","/Arts & Entertainment/Comics & Animation/Cartoons","/Law & Government","/Computers & Electronics/Computer Security","/Jobs & Education/Education/Colleges & Universities","/Computers & Electronics","/Science/Biological Sciences","/Science/Computer Science","/News/Sports News","/Science/Earth Sciences/Geology","/Sports","/Arts & Entertainment/Comics & Animation/Anime & Manga","/Arts & Entertainment/Visual Art & Design","/Computers & Electronics/Consumer Electronics/Game Systems & Consoles","/Arts & Entertainment/Music & Audio/Music Reference","/News/Politics","/Business & Industrial/Aerospace & Defense/Space Technology","/Hobbies & Leisure/Special Occasions/Holidays & Seasonal Events","/Arts & Entertainment/Fun & Trivia","/Arts & Entertainment/Humor","/Arts & Entertainment/Movies","/Online Communities/Social Networks","/Arts & Entertainment/Music & Audio/Pop Music","/Home & Garden","/Jobs & Education/Education","/Games/Computer & Video Games/Casual Games","/Pets & Animals/Wildlife","/Business & Industrial/Agriculture & Forestry","/Computers & Electronics/Computer Hardware","/Games/Computer & Video Games/Simulation Games","/Reference/General Reference","/Science/Astronomy","/News","/Arts & Entertainment/Comics & Animation","/Internet & Telecom/Mobile & Wireless/Mobile & Wireless Accessories"])
N = len(X)

X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()