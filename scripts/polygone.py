# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:16:33 2020

@author: Aksel
"""
import shapely.geometry as sg
import descartes
import pandas as pd
import sklearn.utils as utils
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.path as path
import matplotlib.patches as patches

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