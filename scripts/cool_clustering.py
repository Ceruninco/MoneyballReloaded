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


df = pd.read_csv('../csv/players_stats.csv')

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])
pca = PCA(n_components=0.99, svd_solver = 'full')
pcabis = pca.fit(clustering_df)

reducedDataSet = pcabis.transform(clustering_df)
print(reducedDataSet)
#print(pcabis.explained_variance_ratio_)


# In[3]:


neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(reducedDataSet)
distances, indices = nbrs.kneighbors(reducedDataSet)


# In[4]:


distances = np.sort(distances, axis=0)
distances = distances[:,1]
distancebis = savgol_filter(distances,101,5)
plt.figure(0)
plt.plot(distances)
plt.figure(1)
plt.plot(distancebis)


# compute second derivative
smooth_d1 = np.gradient(distancebis)
smooth_d2 = np.gradient(np.gradient(distancebis))
rotor = Rotor()
new = np.zeros((0,2))
for i in range(0,346):
    array = np.array([[i,distancebis[i]]])
    new = np.append(new, array, axis = 0) 
rotor.fit_rotate(new)
elbow_index = rotor.get_elbow_index()
#print("yop :" +str(elbow_index))
print(new[elbow_index])


plt.figure(2)
plt.plot(smooth_d2)
"""
infls = np.where(np.diff(np.sign(smooth_d2 )))[0]
optiepsiIndex = np.where(smooth_d2 == np.amax(smooth_d2))[0]
optiepsi = distancebis[optiepsiIndex]
print(optiepsi)
"""

# In[5]:


m = DBSCAN(eps=new[elbow_index][1], min_samples=2)
m.fit(reducedDataSet)


# In[6]:


clusters = m.labels_
clusterssorted = np.sort(clusters)


# In[7]:


colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])


# In[8]:

plt.figure(5)
plt.scatter(reducedDataSet[:,0], reducedDataSet[:,1], c=vectorizer(clusters))


# In[9]:


print(len(clusters))
plt.figure(6)

# In[10]:


print(len(clustering_df))


# In[ ]:




