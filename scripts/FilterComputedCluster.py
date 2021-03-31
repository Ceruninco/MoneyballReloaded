# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:49:42 2021

@author: utilisateur
"""
import pandas as pd
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans



###Fonction utilisée pour ouvrir un ComputedCluster et reconstuire un dataframe avec les informations sur les joueurs (équipe,PER, etc..)
def import_cluster(epsilon=0.9,minPoints=2,NoiseProp=0.59):
    pathCluster="../ComputedClusters/"+"epsilon_"+str(epsilon)+"_MinPoints_"+str(minPoints)+"_NoiseProp_"+str(NoiseProp)+".csv"
    df=pd.read_csv(pathCluster,';')
    df.rename(columns={'0':'Player','1':'Cluster'},inplace=True)
    df_PS=pd.read_csv('../csv/players_stats.csv',',')
    df_join=df.join(df_PS.set_index('Player'), on='Player')
    return (df_join)

df=import_cluster()
df_noise=df[df.Cluster==-1] #On garde uniquement le bruit
df_noise=df_noise.reset_index(drop=True)
player_names = pd.DataFrame(df_noise["Player"])
#player_names = player_names.loc[~player_names.index.duplicated(keep='first')]
df = df_noise[['TRB', 'PTS', 'AST', 'DWS', 'TS%', "3PA", "OWS","USG%"]]


fuzzy_kmeans = FuzzyKMeans(k=5, m=2)
fuzzy_kmeans.fit(df)

clusters = pd.DataFrame(fuzzy_kmeans.fuzzy_labels_)
res = pd.concat([player_names, clusters], axis=1)

print('FUZZY_KMEANS')
print(fuzzy_kmeans.cluster_centers_)
    