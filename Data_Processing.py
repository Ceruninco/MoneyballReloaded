# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

df_2016 = pd.read_csv('NBA_totals_2015-2016.csv')
df_2017 = pd.read_csv('NBA_totals_2016-2017.csv')
df_2018 = pd.read_csv('NBA_totals_2017-2018.csv')
df_2019 = pd.read_csv('NBA_totals_2018-2019.csv')
df_2020 = pd.read_csv('NBA_totals_2019-2020.csv')

print(type(df_2020[["Pos"]]))

df_2020[df_2020["Player"] == "Kyle Alexander"]

#displaying only point-gards
pg = df_2020[df_2020["Pos"]=="PG"]

pg.columns

basic_stats_2016 = df_2016.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]
basic_stats_2017 = df_2017.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]
basic_stats_2018 = df_2018.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]
basic_stats_2019 = df_2019.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]
basic_stats_2020 = df_2020.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]


basic_stats = basic_stats_2016.append(basic_stats_2017).append(basic_stats_2018).append(basic_stats_2019).append(basic_stats_2020)

# on group par joueur
summed_basic_stats = basic_stats.groupby(['Player']).sum()


# on enleve ceux qui ont joué moins de 100 matches
summed_basic_stats = summed_basic_stats[summed_basic_stats["G"] > 100 ]

# on arrondi a un chiffre après la virgule
def custom_round_up(x, y):
    return round(x, y)

avg_stats = summed_basic_stats.loc[:,(summed_basic_stats.columns != "Player") & (summed_basic_stats.columns != "G")].div(summed_basic_stats["G"], axis=0)
avg_stats = avg_stats.apply(custom_round_up, args=[1])


# on doit ramener sur 36 minutes
avg_stats_36_minutes = avg_stats.div((avg_stats["MP"]/36) , axis=0)
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[1])
names = pd.DataFrame(avg_stats_36_minutes.index)



# Scaling
avg_stats_36_minutes = avg_stats_36_minutes - avg_stats_36_minutes.min()
avg_stats_36_minutes = avg_stats_36_minutes / ( avg_stats_36_minutes.max() - avg_stats_36_minutes.min() )
avg_stats_36_minutes = avg_stats_36_minutes.apply(custom_round_up, args=[2])
avg_stats_36_minutes_scaled = avg_stats_36_minutes.drop(columns=["MP"])


# on recupere les stats avancées
ad_2016 = pd.read_csv('NBA_advanced_2015-2016.csv')
ad_2017 = pd.read_csv('NBA_advanced_2016-2017.csv')
ad_2018 = pd.read_csv('NBA_advanced_2017-2018.csv')
ad_2019 = pd.read_csv('NBA_advanced_2018-2019.csv')
ad_2020 = pd.read_csv('NBA_advanced_2019-2020.csv')


ad_2016 = ad_2016.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"] ]
ad_2017 = ad_2017.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"] ]
ad_2018 = ad_2018.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"] ]
ad_2019 = ad_2019.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"] ]
ad_2020 = ad_2020.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"] ]


# pour les stats avancées on a besoin de pondérer les stats d'une saison par le nb de matches joués
def ponderateByGamesPlayed(df):
    #On recupere les noms, minutes jouées et matches joués
    names = df["Player"]
    minutes = df["MP"]
    games = df["G"]
    
    # on enleve les noms, minutes jouées et matches joués
    df = df.drop(columns=["Player", "MP", "G"])
    
    # on multiplie chaque stats de chaque joueur par le nb de matches joués pendant cette saison
    df = df.mul(games, axis=0)
    
    # on rajoute les noms, les minutes et des matches joués
    res = pd.concat([names, games, minutes, df], axis=1)
    
    # on rajoute le nom des colonnes
    res.columns = ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]
    return res


ad_2016 = ponderateByGamesPlayed(ad_2016)
ad_2017 = ponderateByGamesPlayed(ad_2017)
ad_2018 = ponderateByGamesPlayed(ad_2018)
ad_2019 = ponderateByGamesPlayed(ad_2019)
ad_2020 = ponderateByGamesPlayed(ad_2020)


# on concat les stats sur les 5 dernieres saisons avant de les aggréger par joueur
summed_ad = ad_2016.append(ad_2017).append(ad_2018).append(ad_2019).append(ad_2020)


# On agrege
agr = {'MP':['sum'],'G': ['sum'], 'PER':['sum'],'TS%':['sum'],'3PAr':['sum'],'TRB%':['sum'],'USG%':['sum'], 'OWS':['sum'], 'DWS':['sum']}
agg_advanced =summed_ad.groupby(['Player']).agg(agr)

# on enleve ceux qui ont joué moins de 100 matches
agg_advanced = agg_advanced[agg_advanced["G"]['sum'] > 100 ]


# on ramene les stats par matches
games = agg_advanced["G"]["sum"]
final_advanced = agg_advanced.div((games) , axis=0)
final_advanced = final_advanced.drop(columns=["G"])
final_advanced = final_advanced.apply(custom_round_up, args=[2])
final_advanced = pd.concat([games, final_advanced], axis=1)
final_advanced.columns = ["G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]


# Scaling
final_advanced_scaled = final_advanced - final_advanced.min()
final_advanced_scaled = final_advanced_scaled / ( final_advanced_scaled.max() - final_advanced_scaled.min() )


final = pd.merge(final_advanced_scaled, avg_stats_36_minutes_scaled, on="Player")



