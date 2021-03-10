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

# keep index while for-loop iterating

for index, item in enumerate(array):
	print(index)
	print(item)


# get the number of rows from df

len(df.index)



 #########################################################
 ############# MACHINE LEARNING WITH SKLEARN #############
 #########################################################


# to build a decision tree regressor with sklearn
# without max_leaves parameter it stops until theres is only one x(i) per leaves

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

# calculate the Mean Square Error

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y, predicted)

# calculate the Mean Absolute Error

from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)

# dict omprehension

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_model = min(scores, key=scores.get)

# use random forest

from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)



# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]


# get the number of missing value per columns

X.isnull().sum()

# get the total number of missing value in the dataset

X.isnull().sum().sum()

# nb of rows with at least a na value

sum([True for idx,row in X_full.iterrows() if any(row.isnull())])


# impute missing values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
# first we fit
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
# then we apply we the fittest value
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# iterate over series

for index, elem in s.items()


# to replace categorical data with binary valie (one extra columns for each possible value)

X = pd.get_dummies(X)




