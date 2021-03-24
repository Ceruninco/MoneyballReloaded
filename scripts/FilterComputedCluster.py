# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 10:49:42 2021

@author: utilisateur
"""
import pandas as pd



###Fonction utilisé pour ouvrir un ComputedCluster et reconstuire un dataframe avec les informations sur les joueurs (équipe,PER, etc..)
def import_cluster(epsilon=0.9,minPoints=2,NoiseProp=0.59):
    pathCluster="../ComputedClusters/"+"epsilon_"+str(epsilon)+"_MinPoints_"+str(minPoints)+"_NoiseProp_"+str(NoiseProp)+".csv"
    df=pd.read_csv(pathCluster,';')
    df.rename(columns={'0':'Player','1':'Cluster'},inplace=True)
    df_PS=pd.read_csv('./players_stats.csv',',')
    df_join=df.join(df_PS.set_index('Player'), on='Player')
    return (df_PS)

df_cluster=import_cluster()
    