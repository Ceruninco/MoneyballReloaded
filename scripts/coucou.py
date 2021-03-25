import pandas as pd
from unidecode import unidecode

csv_files_location = "./csv/"

# on recupere les stats de base
df_2020 = pd.read_csv(csv_files_location+'NBA_totals_2019-2020.csv')
    
# on enleve les accents et caractères spéciaux du nom des joueurs pour les grouper
df_2020["Player"] = df_2020["Player"].apply(unidecode)


# on recupere l'équipe finale de chaque joueur de cette année
# on fait d'une pierre deux coups en récuperant les noms et en filtrant les joueurs ayant pris leur retraite
# avant la saison 2020
team_and_player = df_2020[["Player", "Tm", 'Pos']]
team_and_player = team_and_player.drop_duplicates(subset=['Player'])


# on ne garde que les colonnes qui nous interesse
basic_stats_2020 = df_2020.loc[:, ['Player', 'G', 'MP', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'] ]

# on enleve ceux qui ont joué moins de 100 matches
basic_stats = basic_stats_2020 #[basic_stats_2020["G"] > 100 ]

# on arrondi a un chiffre après la virgule
def custom_round_up(x, y):
    return round(x, y)

avg_stats = basic_stats.loc[:,(basic_stats.columns != "Player") & (basic_stats.columns != "G")].div(basic_stats["G"], axis=0)
avg_stats = avg_stats.apply(custom_round_up, args=[1])


# on recupere les stats avancées
ad_2020 = pd.read_csv(csv_files_location+'NBA_advanced_2019-2020.csv')

# on enleve les accents et caractères spéciaux du nom des joueurs pour les grouper
ad_2020["Player"] = ad_2020["Player"].apply(unidecode)


# on ne garde que les colonnes qui nous intéresse
ad_2020 = ad_2020.loc[:, ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%","TOV%","ORB%"] ]


# on enleve ceux qui ont joué moins de 100 matches => c'est pour cela qu'on a moins de joueurs à la fin!!!!
#agg_advanced = ad_2020[ad_2020["G"] > 100 ]


# on fusionne les stats avancées, les stats de base et les noms des joueurs
final = pd.merge(team_and_player, ad_2020, on="Player")
final = pd.merge(final, basic_stats, on="Player")
print(final)


# we remove the mean and scale to unit variance
#final = StandardScaler().fit_transform(final)


#export to csv
final.to_csv("./csv/players_stats_reloaded.csv")