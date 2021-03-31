#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import sklearn 
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
#from yellowbrick.cluster import KElbowVisualizer
from kneebow.rotor import Rotor
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
sns.set()


# In[2]:


df = pd.read_csv('../csv/players_stats.csv')

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])

results = pd.DataFrame(data = None, columns = ['epsilon' , 'min_size', 'score'],dtype=np.float64) 

for var_portion in np.arange(start = 0.80,stop=0.95,step=0.05,dtype=np.float64):
    print(var_portion)
    pca = PCA(n_components=var_portion, svd_solver = 'full')
    pcabis = pca.fit(clustering_df)

    reducedDataSet = pcabis.transform(clustering_df)


    for eps in np.arange(start = 0.05,stop=0.95,step=0.01,dtype=np.float64):
        for size in np.arange(start = 2,stop=10,step=1,dtype=np.float64): 
            m = DBSCAN(eps=eps, min_samples=size)
            m.fit(reducedDataSet)
            if(max(m.labels_)>1):
                score = sklearn.metrics.silhouette_score(clustering_df,m.labels_)

                results = results.append({'var_portion' : var_portion, 'epsilon' : eps , 'min_size' : size , 'score' : score, 'nb_clusters' : max(m.labels_)+1}, ignore_index=True)

results = results.sort_values(by=[ "nb_clusters"], ascending = False)
results.to_csv("../csv/silhouette_search.csv", sep =';')

# %%

pca = PCA(n_components=0.9, svd_solver = 'full')
pcabis = pca.fit(clustering_df)
dataSet = pcabis.transform(clustering_df)
model = DBSCAN(eps=0.60, min_samples=2)
model.fit(dataSet)
result = pcabis.inverse_transform(dataSet)
print(model.labels_)

# %%

