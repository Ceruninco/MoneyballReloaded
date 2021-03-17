# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:16:33 2020

@author: Aksel
"""
import shapely.geometry as sg
import descartes
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
import seaborn as sns
import chart_studio.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
from plotly.graph_objs import *

"""
Ad1 =  pd.read_csv('NBA_advanced_2019-2020.csv',delimiter = ',')
Ad1.insert(2,'Year',2020)
Ad2=pd.read_csv('NBA_advanced_2018-2019.csv',delimiter = ',')
Ad2.insert(2,'Year',2019)
Ad3=pd.read_csv('NBA_advanced_2017-2018.csv',delimiter = ',')
Ad3.insert(2,'Year',2018)
Ad4=pd.read_csv('NBA_advanced_2016-2017.csv',delimiter = ',')
Ad4.insert(2,'Year',2017)
"""
PlayerStats="MP"

NormalizeData = pd.read_csv("../csv/players_stats.csv", delimiter =",");

#Ad = Ad1.append(Ad2).append(Ad3).append(Ad4)


""" a function which computes a performance polygon for a specific player using three parameters
 (Offensive Win Share, Defensive Win Share, Win Share)"""

def performance_polygon(PlayerName):
    Player=10*NormalizeData[NormalizeData.Player.eq(PlayerName)]

    # Player = AdDisp[AdDisp.Year.eq(2020)]

    properties = ['Offensive Win share', 'Defensive win share', 'AST','TS%', "TRB%", "PTS", "3PA", ]
    values = np.random.uniform(5,9,len(properties))

    values = [Player['OWS'], Player['DWS'], Player['AST'], Player["TS%"], Player["TRB%"], Player["PTS"], Player["3PA"]]
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
    """
    maxi = max([Player.iloc[0,19]+1, Player.iloc[0,20]+1, Player.iloc[0,21]+1])
    if maxi < 10:
        plt.ylim(0,10)
    else:
        plt.ylim(0,maxi)
        """
    plt.ylim(0,10)
    for i in range(len(properties)):
        angle_rad = i/float(len(properties))*2*np.pi
        angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 10.75, properties[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

    plt.title("Statistics of "+PlayerName)
    plt.show()
    
def performance_polygon_vs_player(*PlayerName):
    colors = ["blue", "red", "green", "orange", "brown"]
    fig = plt.figure(figsize=(10,8), facecolor='white')
    for i in range (0,len(PlayerName)):
        Player=10*NormalizeData[NormalizeData.Player.eq(PlayerName[i])]
        #Player2=10*NormalizeData[NormalizeData.Player.eq(PlayerName[1])]
        
        # Player = AdDisp[AdDisp.Year.eq(2020)]
    
        properties = ['Offensive Win share', 'Defensive win share', 'AST','TS%', "TRB", "PTS", "3PA", ]
        values = np.random.uniform(5,9,len(properties))
    
        values1 = [Player['OWS'], Player['DWS'], Player['AST'], Player["TS%"], Player["TRB"], Player["PTS"], Player["3PA"]]
        #values2 = [Player2['OWS'], Player2['DWS'], Player2['AST'], Player2["TS%"], Player2["TRB%"], Player2["PTS"], Player2["3PA"]]
        matplotlib.rc('axes', facecolor = 'white')
    
        
    
        axes = plt.subplot(111, polar=True)
    
        t = np.arange(0,2*np.pi,2*np.pi/len(properties))
        plt.xticks(t, [])
    
        points = [(x,y) for x,y in zip(t,values1)]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO,] + \
                [path.Path.LINETO,]*(len(values) -1) + \
                [ path.Path.CLOSEPOLY ]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=False, color=colors[i], linewidth=0, alpha=.2)
        axes.add_patch(_patch)
        _patch = patches.PathPatch(_path, fill=False, edgecolor=colors[i], linewidth = 2, label=PlayerName[i])
        axes.add_patch(_patch)
        plt.scatter(points[:,0],points[:,1], linewidth=2,
                s=50, color='white', edgecolor='black', zorder=10)
    #plt.scatter(points[:,0],points[:,1], linewidth=2,s=50, color='white', edgecolor='black', zorder=10)
    plt.legend(loc="lower right",borderaxespad=-6)
    """
    maxi = max([Player.iloc[0,19]+1, Player.iloc[0,20]+1, Player.iloc[0,21]+1])
    if maxi < 10:
        plt.ylim(0,10)
    else:
        plt.ylim(0,maxi)
        """
    plt.ylim(0,10)
    for i in range(len(properties)):
        angle_rad = i/float(len(properties))*2*np.pi
        angle_deg = i/float(len(properties))*360
        ha = "right"
        if angle_rad < np.pi/2 or angle_rad > 3*np.pi/2: ha = "left"
        plt.text(angle_rad, 10.75, properties[i], size=14,
                 horizontalalignment=ha, verticalalignment="center")

    plt.title("Performance polygon", pad = 50)
    plt.show()



PlayerName='LeBron James'
#PlayerName5= "Buddy Hield"
PlayerName6="Stephen Curry"
#performance_polygon(PlayerName)
PlayerName2 = "Aaron Brooks"
PlayerName3 = "Brandon Knight"
PlayerName4 = "Jamal Crawford"

performance_polygon_vs_player(PlayerName, PlayerName6, PlayerName2, PlayerName3, PlayerName4)


def histogram_minutes_played_random_chosen_players():
    shuffledData = utils.shuffle(Ad)
    sample = shuffledData[1:20]
    #sample.plot.bar(x='Player', y='G', rot=90)
"""

histogram_minutes_played_random_chosen_players()

Ad_G_MP = Ad1[['G','MP']]
#clustering = cluster.DBSCAN(eps=5, min_samples=10).fit_predict(Ad_G_MP)
plt.scatter(Ad_G_MP['MP'],Ad_G_MP['G'])
plt.figure()


Ad_MP_3P = Ad1[['MP','3P']]
#clustering = cluster.DBSCAN(eps=5, min_samples=10).fit_predict(Ad_MP_3P)
plt.scatter(Ad_MP_3P['MP'],Ad_MP_3P['3P'])
plt.figure()


df = pd.read_csv('NBA_totals_2019-2020.csv')
df[['FG%', '3P%', '2P%', 'FT%', 'eFG%']] = \
df[['FG%', '3P%', '2P%', 'FT%', 'eFG%']].fillna(value=0)
df['nPTS'] = df['PTS']/df['MP']
df['nPF'] = df['PF']/df['MP']
df['nTOV'] = df['TOV']/df['MP']
df['nBLK'] = df['BLK']/df['MP']
df['nSTL'] = df['STL']/df['MP']
df['nAST'] = df['AST']/df['MP']
reducedDS = df[['Player','nPTS','nPF','nTOV','nBLK','nSTL','nAST','FG%','3P%','2P%','FT%']]

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
plt.scatter(reducedDS['nPTS'], reducedDS['FT%'], c=None, **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(True)
frame.axes.get_yaxis().set_visible(True)

def plot_clusters(data, algorithm, args, kwds):
    labels = algorithm(*args, **kwds).fit_predict(data)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data[data.columns[0]], data[data.columns[1]], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(True)
    frame.axes.get_yaxis().set_visible(True)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    
test = reducedDS.loc[:, reducedDS.columns != 'Player']
plot_clusters(test[['nPTS','nAST']], cluster.DBSCAN, (), {'eps':0.01})
test[test.columns[0]]


init_notebook_mode()
df = pd.read_csv('alpha_shape.csv')
df.head()

point = dict(
    mode = "markers",
    name = "y",
    type = "scatter3d",
    x = Ad1['DWS'], y = Ad1['DRB%'], z = Ad1['DBPM'], text=Ad1["Player"],
    marker = dict( size=2, color="rgb(23, 190, 207)" )
)

mesh = dict(
    alphahull = 50,
    name = "y",
    opacity = 0.1,
    type = "mesh3d",
    x = Ad1['DWS'], y = Ad1['DRB%'], z = Ad1['DBPM']
)
layout = dict(
    title = '3d representation',
    scene = dict(
        xaxis = dict( zeroline=False, title ="DWS" ),
        yaxis = dict( zeroline=False, title = "DRB%" ),
        zaxis = dict( zeroline=False, title = "DBPM" ),
    )
)

fig = dict( data=[point, mesh],  layout=layout )

plot(fig,filename='clustering 3D')

"""
