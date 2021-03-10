# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:22:52 2021

@author: hugod
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NBA_totals_2019-2020.csv')
#print(df.shape)
df = df.dropna()
print(df)
data = np.zeros((df.shape[0],3))
#print(df[['2P%', '3P%']].values[0])

for i in range(df.shape[0]):
    data[i] = df[['TRB', 'PTS', 'AST']].values[i]
    
print(data)
X = StandardScaler().fit_transform(data)

#print(X)

# Compute DBSCAN
db = DBSCAN(eps=0.1, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print(labels)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)


res = np.zeros((0,2))
result = pd.DataFrame(res)
for i in range(df.shape[0]):
    row = [[df['Player'].values[i], labels[i]]]
    result = result.append(row)



print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
#print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
#print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
#print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


# White removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # White used for noise.
        col = [1, 1, 1, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.xlabel("TRB")
#plt.ylabel("PTS")

plt.show()