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
"""
for var_portion in np.arange(start = 0.6,stop=0.95,step=0.05,dtype=np.float64):
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
"""
# %%

pca = PCA(n_components=0.85, svd_solver = 'full')
pcabis = pca.fit(clustering_df)
dataSet = pcabis.transform(clustering_df)
model = DBSCAN(eps=0.22, min_samples=2)
model.fit(dataSet)
result = pcabis.inverse_transform(dataSet)
res = np.zeros((0,3))
cluster = pd.DataFrame(res)
for k in range(df.shape[0]):
    row = [[df['Player'].values[k], model.labels_[k], df["Pos"].values[k]]]
    cluster = cluster.append(row)
    cluster = cluster.sort_values(by=[1], ascending = False)
noise_number = 0
value = cluster[1].values[0] 
compteur = 0
pseudo_clust = 0 
for i in range(df.shape[0]):
    if (value == cluster[1].values[i] and  cluster[1].values[i] != -1):
        compteur += 1
    else:
        if compteur > 5:
            noise_number += compteur
            print(str(value) + " : " + str(compteur))
            pseudo_clust += compteur
        compteur = 1
    value =  cluster[1].values[i]

    if  cluster[1].values[i] == -1:
        noise_number += 1
        
print("nombre de points bruités : " +str(noise_number)+" et pseudo_clust = "+str(pseudo_clust))
print(model.labels_)

# %%

