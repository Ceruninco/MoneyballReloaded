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


"""
df = pd.read_csv('NBA_totals_2019-2020.csv')


df[['FG%', '3P%', '2P%', 'FT%', 'eFG%']] = \
df[['FG%', '3P%', '2P%', 'FT%', 'eFG%']].fillna(value=0)
print(df)

print(df[['PTS','PF','TOV','BLK','STL','AST']])

df['nPTS'] = df['PTS']/df['MP']
df['nPF'] = df['PF']/df['MP']
df['nTOV'] = df['TOV']/df['MP']
df['nBLK'] = df['BLK']/df['MP']
df['nSTL'] = df['STL']/df['MP']
df['nAST'] = df['AST']/df['MP']

reducedDS = df[['Player','nPTS','nPF','nTOV','nBLK','nSTL','nAST','FG%','3P%','2P%','FT%']]

print(reducedDS)
%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

plt.scatter(reducedDS['nPTS'], reducedDS['FT%'], c=None, **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)
"""

"""
def plot_clusters(data, algorithm, args, kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data[data.columns[0]], data[data.columns[1]], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.plot()
    
test = reducedDS.loc[:, reducedDS.columns != 'Player']
plot_clusters(test[['nPTS','nAST']], cluster.DBSCAN, (), {'eps':0.01})

test[test.columns[0]]
"""

"""
X = pd.DataFrame([avg_stats_36_minutes_scaled['PTS'], avg_stats_36_minutes_scaled['TRB'], avg_stats_36_minutes_scaled['AST'] ])
X_t = X.transpose()
print(X_t)
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

frame = plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

def plot_clusters(data, algorithm, args, kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    print(labels)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data[data.columns[0]], data[data.columns[1]], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
   

plot_clusters(X_t, cluster.DBSCAN, (), {'eps':0.05})
"""

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