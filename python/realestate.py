import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
dataframe = pandas.read_csv("train.csv")

numericColumns=['MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'MasVnrArea', 'BsmtFinSF1'
                          ,'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','X1stFlrSF','X2ndFlrSF'
                          ,'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'
                          ,'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea'
                          ,"GarageYrBlt", "YrSold", "YearBuilt",    "YearRemodAdd" , "MiscVal", "MoSold", "X3SsnPorch", "ScreenPorch", "PoolArea"
                          ,"WoodDeckSF","OpenPorchSF","EnclosedPorch", "LowQualFinSF"
                          ]
numerics = dataframe.loc[:,numericColumns]

# split into input (X) and output (Y) variables
Y = dataframe.loc[:,'SalePrice'].values
X = numerics.values


# Make a multi layer model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(35, input_dim=35, init='normal', activation='relu'))
	model.add(Dense(17, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


numpy.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, nb_epoch=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


