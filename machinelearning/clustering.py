from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import math
import pandas as pd

k_function = lambda N: int(round(math.sqrt(N)))

def train_clustering(df):
    """
    Trains the pipeline for performing the clustering of the video feature
    vectors.
    """

    #SET CSV FILE HERE
    #PROVIDE THE CSV HEADER
    X = df.as_matrix(["sentiment","/Science/Computer Science","/Jobs & Education/Education","/Science/Mathematics","/Arts & Entertainment/TV & Video/Online Video","/Arts & Entertainment/Music & Audio/World Music","/Arts & Entertainment/Music & Audio/Music Streams & Downloads","/Games","/Science/Physics","/Arts & Entertainment/Music & Audio/Urban & Hip-Hop","/Sports","/Food & Drink/Cooking & Recipes","/Arts & Entertainment/Comics & Animation/Anime & Manga","/Computers & Electronics/Software/Multimedia Software","/Arts & Entertainment/TV & Video","/Computers & Electronics/Consumer Electronics/Game Systems & Consoles","/Arts & Entertainment/Online Media","/Hobbies & Leisure","/News/Sports News","/Computers & Electronics/Computer Hardware/Computer Drives & Storage","/Home & Garden/Home Appliances","/Reference/General Reference/Calculators & Reference Tools","/Arts & Entertainment/Music & Audio/Music Reference","/Arts & Entertainment/Music & Audio/Soundtracks","/Computers & Electronics/Software","/Arts & Entertainment/Music & Audio/Rock Music","/Hobbies & Leisure/Special Occasions/Holidays & Seasonal Events","/Arts & Entertainment/Music & Audio/Music Equipment & Technology","/Games/Roleplaying Games","/Arts & Entertainment","/Arts & Entertainment/Music & Audio/Dance & Electronic Music","/Adult","/Internet & Telecom","/Business & Industrial/Agriculture & Forestry","/Science/Earth Sciences","/Autos & Vehicles","/News/Politics","/People & Society/Religion & Belief","/Reference","/Arts & Entertainment/TV & Video/TV Shows & Programs","/Home & Garden/Kitchen & Dining","/Sports/Team Sports/Cricket","/Shopping/Apparel","/Arts & Entertainment/Humor","/Arts & Entertainment/Movies","/Arts & Entertainment/Music & Audio/Radio","/Sports/Individual Sports/Cycling","/Business & Industrial/Business Operations/Management","/Computers & Electronics/Programming","/Arts & Entertainment/Fun & Trivia","/Health/Health Conditions/Infectious Diseases","/Arts & Entertainment/Music & Audio/Pop Music","/Arts & Entertainment/Music & Audio/Jazz & Blues","/Science","/Home & Garden","/Online Communities","/Computers & Electronics/Consumer Electronics","/Arts & Entertainment/Comics & Animation/Comics","/Games/Computer & Video Games/Casual Games","/Finance/Investing/Currencies & Foreign Exchange","/Arts & Entertainment/Music & Audio/Classical Music","/Arts & Entertainment/Music & Audio","/People & Society/Subcultures & Niche Interests","/Books & Literature","/Computers & Electronics/Computer Hardware","/Reference/General Reference","/Law & Government","/People & Society/Social Sciences/Psychology","/Reference/Humanities/History","/Arts & Entertainment/Comics & Animation/Cartoons","/Science/Biological Sciences/Neuroscience","/Online Communities/Blogging Resources & Services","/Science/Astronomy","/Home & Garden/Kitchen & Dining/Small Kitchen Appliances","/Arts & Entertainment/Comics & Animation","/Sports/Team Sports/Baseball","/Games/Computer & Video Games","/Science/Mathematics/Statistics","/Jobs & Education/Education/Colleges & Universities","/Computers & Electronics","/Arts & Entertainment/Fun & Trivia/Flash-Based Entertainment","/Science/Biological Sciences","/Games/Online Games/Massively Multiplayer Games"
])
    X = np.nan_to_num(X)
    N = len(X)

    #ISOMAP algorithm w/ M equal to 2D
    X_isomap = Isomap(n_components=2)
    X_mapped_matrix = X_isomap.fit_transform(X)

    #Define k clusters as the sqrt of the total amount of samples
    k = k_function(N)

    #KMeans w/ k clusters
    kmeans_object = KMeans(n_clusters=k, random_state=0)
    #Obtain the labels that map the data to their cluster
    kmeans_labels = kmeans_object.fit(X_mapped_matrix)
    cluster_labels = kmeans_labels.labels_
    #Transform
    kmeans_cluster = kmeans_object.transform(X_mapped_matrix)

    #Scale the colors for the clusters proportional to 256 / the number clusters
    for i in cluster_labels:
            i = i * (256 / len(cluster_labels))

    x = kmeans_cluster[:, 0]
    y = kmeans_cluster[:, 1]

    return (X_isomap, kmeans_object)

def predict(data, model, vector):
    X_isomap = model[0]
    kmeans_object = model[1]
    kmeans_labels = kmeans_object
    cluster_labels = kmeans_labels.labels_

    #Prediction w/ arbitrary vector
    arb_vect = vector
    #Transform the vector & perform a prediction to get the most similar cluster
    transformed_arb_vect = X_isomap.transform(np.expand_dims(arb_vect, axis=0))
    prediction_cluster = kmeans_labels.predict(transformed_arb_vect)

    #Create a list of indicies
    indicies = []
    for i in range(0, len(cluster_labels), 1):
            if cluster_labels[i] == prediction_cluster[0]:
                    indicies.append(i)

    #Create a list of video ids corresponding to all in the cluster that is most similar to the provided vector
    video_ids = []
    for i in indicies:
            video_ids.append(data["video_id"].iloc[i])
    #Print the video Ids
    return video_ids

    #Plot
    #plt.scatter(x, y, s=50, c=cluster_labels, alpha=0.5)
    #plt.show()
