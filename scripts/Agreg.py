# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:16:33 2020

@author: Aksel
"""

import pandas as pd


Ad1 =  pd.read_csv('NBA_advanced_2019-2020.csv',delimiter = ',')
Ad1.insert(2,'Year',2020)
Ad2=pd.read_csv('NBA_advanced_2018-2019.csv',delimiter = ',')
Ad2.insert(2,'Year',2019)
Ad3=pd.read_csv('NBA_advanced_2017-2018.csv',delimiter = ',')
Ad3.insert(2,'Year',2018)
Ad4=pd.read_csv('NBA_advanced_2016-2017.csv',delimiter = ',')
Ad4.insert(2,'Year',2017)  
Ad5=pd.read_csv('NBA_advanced_2015-2016.csv',delimiter = ',')
Ad5.insert(2,'Year',2016)
total=pd.read_csv('NBA_totals_2019-2020.csv',delimiter = ',')


def SupDuplicateTeam(Ad):
    AdSort=Ad.sort_values(by=['Player'],ascending=False)
    index=[]
    index=AdSort.iloc[AdSort['Tm'].loc[AdSort['Tm']=="TOT"].index]['Tm']
    
    AdSort.loc[AdSort['Tm']=="TOT", ['Tm']]='Transfert'
    
    AdDrop=AdSort.drop_duplicates(subset='Player', keep='first')
    
    #Adtest=Ad.groupby(['Player','G']).agg('max')
    return(AdDrop)

Ad1=SupDuplicateTeam(Ad1)
Ad2=SupDuplicateTeam(Ad2)
Ad3=SupDuplicateTeam(Ad3)
Ad4=SupDuplicateTeam(Ad4)
Ad5=SupDuplicateTeam(Ad5)
Ad = Ad1.append(Ad2).append(Ad3).append(Ad4).append(Ad5)




def avgAdvanced(Ad):
    agr={'Year':['min','max'],'Tm':['max'],'G':['sum','mean'],'MP':['sum','mean'],'PER':['mean'],'TS%':['mean'],'3PAr':['mean'],'FTr':['mean'],'ORB%':['mean']}
    Avg1=Ad.groupby(['Player']).agg(agr)
    Avg1['PER']['mean'].apply(lambda x: x*Avg1['G']['mean'])
    
    return(Avg1)


Agr=avgAdvanced(Ad)




