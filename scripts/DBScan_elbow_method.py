# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


# %%
df = pd.read_csv('./csv/players_stats.csv')

clustering_df = df.drop(columns=["Unnamed: 0","Player", "final_team","Pos"])


# %%
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(clustering_df)
distances, indices = nbrs.kneighbors(clustering_df)


# %%
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)


# %%
m = DBSCAN(eps=0.6, min_samples=5)
m.fit(clustering_df)


# %%
clusters = m.labels_


# %%
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])


# %%
plt.scatter(clustering_df[clustering_df.columns[0]], clustering_df[clustering_df.columns[1]], c=vectorizer(clusters))


# %%



