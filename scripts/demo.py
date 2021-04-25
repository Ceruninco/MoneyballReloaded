#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:20:27 2021

@author: elie
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import pandas as pd
import numpy as np
from Dissimilarity_Matrix import get_most_similar_players, plot_heat_matrix, get_distance_between_players
from polygone import performance_polygon_vs_player
import matplotlib.image as mpimg

stats = pd.read_csv("../csv/players_stats.csv")
dist_mat = pd.read_csv("../csv/distance_matrix.csv")
dist_mat = dist_mat.rename(columns={"Unnamed: 0": 'Name'})
criterias = ['OWS', 'DWS', 'AST','TS%', "TRB", "PTS", "3PA" ]



st.title('MoneyBall Reloaded')

player = st.selectbox(
    'Which player do you want to find similar players to?',
     stats['Player'])

'You selected: ', player

number = st.selectbox(
    'How many players do you want among the most similar?',
     np.arange(1, 10, 1))

'You selected:', number


# get the n most similar to the required player
most_similar_players = get_most_similar_players(player, number, dist_mat)
most_similar_players_names = [names for (names, score) in most_similar_players ]

most_similar_players_names.append(player)

# get the distance between the n most similar players of the required player
players_distances = get_distance_between_players(most_similar_players_names, dist_mat)
only_number_matrix = [list(value.values()) for key, value in players_distances.items()]

# transform to proper df
df_most_similar_players = pd.DataFrame(most_similar_players)
df_most_similar_players.columns = ["Name", "Similarity"]
df_most_similar_players.set_index('Name', inplace=True)


# get the heat matrix of the n most similar players
heat_matrix = plot_heat_matrix(only_number_matrix, most_similar_players_names)

# draw polygones for the n most similar players
players_to_draw = [player[0] for player in most_similar_players]
players_to_draw.append(player)
polygones = performance_polygon_vs_player(players_to_draw, criterias)

heat_matrix.savefig("heat_matrix.jpg")
polygones.savefig("polygones.jpg")


# Display
st.table(df_most_similar_players)
#st.write(heat_matrix)
#st.write(polygones)


heat_matrix_image = mpimg.imread('heat_matrix.jpg')
polygones_image = mpimg.imread("polygones.jpg")

col1, col2 = st.beta_columns(2)

col1.header("Polygones")
col1.image(polygones_image, use_column_width=True)

col2.header("Heat Matrix")
col2.image(heat_matrix_image, use_column_width=True)

