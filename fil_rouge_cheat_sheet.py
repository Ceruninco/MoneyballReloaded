# pour multiplier/div un df par une ligne/colonne

final_advanced = agg_ad.div((games) , axis=0)
df = df.mul(games, axis=0)

# pour drop une colonne

final_advanced = final_advanced.drop(columns=["G"])


#Pour concat des df en ligne ou en colonne

final_advanced = pd.concat([final_advanced, games], axis=1)


#Pour aggreger des données avec des fonctions différentes selon la colonne

agr = {'MP':['sum'],'G': ['sum'], 'PER':['sum'],'TS%':['sum'],'3PAr':['sum'],'TRB%':['sum'],'USG%':['sum'], 'OWS':['sum'], 'DWS':['sum']}

agg_ad =summed_ad.groupby(['Player']).agg(agr)


#Pour appliquer une fonction a tous les éléments d'un df

final_advanced = final_advanced.apply(custom_round_up, args=[2])

# pour renommer les colonnes d'un df

res.columns = ["Player", "G", "MP", "PER", "TS%", "3PAr", "TRB%", "USG%", "OWS", "DWS"]


# pour ne garder que certaines col d'un df => reste des df et non des séries

df = df.loc[:,(df.columns != "Player") & (df.columns != "G")]

df = df[["NomCol1", "NomCol2"]]

df = df.NomCol1

# filtrer un df par rapport aux valeurs des colonnes

df = df[df["G"] > 100 ]

# pour concat en ligne des df ayant les memes colonnes

df = df.append(df_2).append(df_3).append(df_4).append(df_5)

# Pour scaler les donnes 

final_advanced = final_advanced - final_advanced.min()
final_advanced = final_advanced / ( final_advanced.max() - final_advanced.min() )
final_advanced


# Pour transformer un np array en df

df = pd.DataFrame(np_arr)

# pour obtenir des stats de bases sur les colonnes

df.describre()

# pour avoir les premieres lignes

df.head()

# To print lines which fulfil a column condition

df_2020[df_2020["Player"] == "Kyle Alexander"]




 #########################################################
 # MACHINE LEARNING WITH SKLEARN #########################
 #########################################################


# to build a decision tree regressor with sklearn

from sklearn.tree import DecisionTreeRegressor  
#specify the model. 
model = DecisionTreeRegressor(random_state=0)

# Fit the model
model.fit(X, y)

# Predict => return array-like of predicted values

predictions = model.predict(X)


# split the data for validation
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# calculate the MSE

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predicted_home_prices)

#

