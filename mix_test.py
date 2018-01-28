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

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=50, color=color)
		
        # Plot an ellipse to show the Gaussian component
        #angle = np.arctan(u[1] / u[0])
        #angle = 180. * angle / np.pi  # convert to degrees
        #ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        #ell.set_alpha(0.5)
        #splot.add_artist(ell)

    #plt.xlim(-9., 5.)
    #plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

# Number of samples per component
n_samples = 500

#PLACE X VECTOR HERE
#SET CSV FILE HERE
df = pd.read_csv('csv2.csv')
#PROVIDE THE CSV HEADER
X = df.as_matrix(["sentiment","/Science/Mathematics","/Science/Physics","/News/Business News","/Home & Garden/Home Appliances","/Science/Earth Sciences","/Reference","/Arts & Entertainment/Music & Audio/Radio","/Science","/Finance/Investing/Currencies & Foreign Exchange","/People & Society/Subcultures & Niche Interests","/Sports/Team Sports","/Reference/Humanities/History","/Health/Public Health","/Law & Government/Public Safety/Law Enforcement","/Sports/Team Sports/Baseball","/Games/Computer & Video Games","/Arts & Entertainment/Fun & Trivia/Flash-Based Entertainment","/Games/Computer & Video Games/Shooter Games","/Home & Garden/Kitchen & Dining","/Computers & Electronics/Software","/Arts & Entertainment/Music & Audio/World Music","/Games","/Sensitive Subjects","/Arts & Entertainment/TV & Video","/Reference/General Reference/Calculators & Reference Tools","/Business & Industrial/Energy & Utilities","/Computers & Electronics/Consumer Electronics","/Games/Roleplaying Games","/Arts & Entertainment","/Arts & Entertainment/Music & Audio/Dance & Electronic Music","/Arts & Entertainment/Music & Audio/Religious Music","/Arts & Entertainment/TV & Video/TV Shows & Programs","/Sports/Team Sports/Cricket","/Jobs & Education/Education/Teaching & Classroom Resources","/Arts & Entertainment/Music & Audio","/Law & Government/Public Safety","/Science/Engineering & Technology","/Food & Drink","/Arts & Entertainment/Performing Arts","/Home & Garden/Kitchen & Dining/Small Kitchen Appliances","/Arts & Entertainment/TV & Video/Online Video","/Business & Industrial","/Law & Government/Public Safety/Crime & Justice","/Arts & Entertainment/Music & Audio/Urban & Hip-Hop","/Computers & Electronics/Software/Multimedia Software","/Arts & Entertainment/Online Media","/Hobbies & Leisure","/Arts & Entertainment/Music & Audio/Music Equipment & Technology","/Arts & Entertainment/Music & Audio/Soundtracks","/Arts & Entertainment/Music & Audio/Rock Music","/People & Society/Religion & Belief","/Computers & Electronics/Software/Business & Productivity Software","/Books & Literature","/Computers & Electronics/CAD & CAM","/Games/Board Games/Chess & Abstract Strategy Games","/Internet & Telecom","/Health/Health Conditions/Infectious Diseases","/Online Communities","/Arts & Entertainment/Comics & Animation/Comics","/Business & Industrial/Energy & Utilities/Renewable & Alternative Energy","/Arts & Entertainment/Comics & Animation/Cartoons","/Law & Government","/Computers & Electronics/Computer Security","/Jobs & Education/Education/Colleges & Universities","/Computers & Electronics","/Science/Biological Sciences","/Science/Computer Science","/News/Sports News","/Science/Earth Sciences/Geology","/Sports","/Arts & Entertainment/Comics & Animation/Anime & Manga","/Arts & Entertainment/Visual Art & Design","/Computers & Electronics/Consumer Electronics/Game Systems & Consoles","/Arts & Entertainment/Music & Audio/Music Reference","/News/Politics","/Business & Industrial/Aerospace & Defense/Space Technology","/Hobbies & Leisure/Special Occasions/Holidays & Seasonal Events","/Arts & Entertainment/Fun & Trivia","/Arts & Entertainment/Humor","/Arts & Entertainment/Movies","/Online Communities/Social Networks","/Arts & Entertainment/Music & Audio/Pop Music","/Home & Garden","/Jobs & Education/Education","/Games/Computer & Video Games/Casual Games","/Pets & Animals/Wildlife","/Business & Industrial/Agriculture & Forestry","/Computers & Electronics/Computer Hardware","/Games/Computer & Video Games/Simulation Games","/Reference/General Reference","/Science/Astronomy","/News","/Arts & Entertainment/Comics & Animation","/Internet & Telecom/Mobile & Wireless/Mobile & Wireless Accessories"])
N = len(X)

# Fit a Gaussian mixture with EM using five components
gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0, 'Gaussian Mixture')

# Fit a Dirichlet process Gaussian mixture using five components
dpgmm = mixture.BayesianGaussianMixture(n_components=2, covariance_type='full').fit(X)
plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')

plt.show()