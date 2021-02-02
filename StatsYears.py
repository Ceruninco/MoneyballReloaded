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

PlayerName='Alan Anderson'
PlayerStats="MP"

Ad = Ad1.append(Ad2).append(Ad3).append(Ad4)
AdDisp=Ad[Ad.Player.eq(PlayerName)]

#kmeans = KMeans(n_clusters=2, random_state=0).fit(Ad)

AdDisp.plot(x="Year",y=PlayerStats,title=PlayerStats+" of "+PlayerName)



