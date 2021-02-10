# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:16:33 2020

@author: Aksel
"""

import pandas as pd
import sklearn.cluster as cluster
import sklearn.utils as utils
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


Ad1 =  pd.read_csv('NBA_advanced_2019-2020.csv',delimiter = ',')
Ad1.insert(2,'Year',2020)
Ad2=pd.read_csv('NBA_advanced_2018-2019.csv',delimiter = ',')
Ad2.insert(2,'Year',2019)
Ad3=pd.read_csv('NBA_advanced_2017-2018.csv',delimiter = ',')
Ad3.insert(2,'Year',2018)
Ad4=pd.read_csv('NBA_advanced_2016-2017.csv',delimiter = ',')
Ad4.insert(2,'Year',2017)

PlayerStats="MP"

Ad = Ad1.append(Ad2).append(Ad3).append(Ad4)

def performance_polygon(PlayerName):
    AdDisp=Ad[Ad.Player.eq(PlayerName)]

    Player = AdDisp[AdDisp.Year.eq(2020)]

    properties = ['Offensive Win share', 'Defensive win share', 'Win share']
    values = np.random.uniform(5,9,len(properties))

    values = [Player['OWS'], Player['DWS'], Player['WS']]
    matplotlib.rc('axes', facecolor = 'white')

    fig = plt.figure(figsize=(10,8), facecolor='white')

    axes = plt.subplot(111, polar=True)

    t = np.arange(0,2*np.pi,2*np.pi/len(properties))
    plt.xticks(t, [])

    points = [(x,y) for x,y in zip(t,values)]
    points.append(points[0])
    points = np.array(points)
    codes = [path.Path.MOVETO,] + \
            [path.Path.LINETO,]*(len(values) -1) + \
            [ path.Path.CLOSEPOLY ]
    _path = path.Path(points, codes)
    _patch = patches.PathPatch(_path, fill=True, color='blue', linewidth=0, alpha=.1)
    axes.add_patch(_patch)
    _patch = patches.PathPatch(_path, fill=False, linewidth = 2)
    axes.add_patch(_patch)

    plt.scatter(points[:,0],points[:,1], linewidth=2,
                s=50, color='white', edgecolor='black', zorder=10)

    maxi = max([Player.iloc[0,19]+1, Player.iloc[0,20]+1, Player.iloc[0,21]+1])
    if maxi < 10:
        plt.ylim(0,10)
    else:
        plt.ylim(0,maxi)

    for i in range(len(properties)):
        angle_rad = i/float(len(properties))*2*np.pi
        angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 10.75, properties[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

    plt.title("Statistics of "+PlayerName)
    plt.show()


PlayerName='James Harden'
performance_polygon(PlayerName)



#kmeans = KMeans(n_clusters=2, random_state=0).fit(Ad)

#AdDisp.plot(x="Year",y=PlayerStats,title=PlayerStats+" of "+PlayerName)



def histogram_minutes_played_random_chosen_players():
    shuffledData = utils.shuffle(Ad)
    sample = shuffledData[1:20]
    #sample.plot.bar(x='Player', y='G', rot=90)


histogram_minutes_played_random_chosen_players()

Ad_G_MP = Ad1[['G','MP']]
#clustering = cluster.DBSCAN(eps=5, min_samples=10).fit_predict(Ad_G_MP)
plt.scatter(Ad_G_MP['MP'],Ad_G_MP['G'])
plt.figure()

"""
Ad_MP_3P = Ad1[['MP','3P']]
#clustering = cluster.DBSCAN(eps=5, min_samples=10).fit_predict(Ad_MP_3P)
plt.scatter(Ad_MP_3P['MP'],Ad_MP_3P['3P'])
plt.figure()
"""
