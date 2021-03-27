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
sns.set()


# In[2]:


df = pd.read_csv('./csv/players_stats.csv')

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])
pca = PCA(n_components=0.85, svd_solver = 'full')
pcabis = pca.fit(clustering_df)

reducedDataSet = pcabis.transform(clustering_df)


# %%


results = pd.DataFrame(data = None, columns = ['epsilon' , 'min_size', 'score'],dtype=np.float64) 
print('dickhead')
for eps in np.arange(start = 0.05,stop=0.95,step=0.01,dtype=np.float64):
    for size in np.arange(start = 2,stop=10,step=1,dtype=np.float64): 
        m = DBSCAN(eps=eps, min_samples=size)
        m.fit(reducedDataSet)
        if(max(m.labels_)>1):
            score = sklearn.metrics.silhouette_score(reducedDataSet,m.labels_)
            results = results.append({'epsilon' : eps , 'min_size' : size , 'score' : score}, ignore_index=True)


results.to_csv("./csv/silhouette_search.csv")