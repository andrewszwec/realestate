import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as pp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import linear_model

## Make some helper functions
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# load dataset
dataframe = pd.read_csv("/Users/andrew/Documents/kaggle/realestate/data/train.csv")
submission = pd.read_csv("/Users/andrew/Documents/kaggle/realestate/data/test.csv")


## Deal with Training Set
numericColumns=['MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'MasVnrArea', 'BsmtFinSF1'
                          ,'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','X1stFlrSF','X2ndFlrSF'
                          ,'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'
                          ,'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea'
                          ,"GarageYrBlt", "YrSold", "YearBuilt",    "YearRemodAdd" , "MiscVal", "MoSold", "X3SsnPorch", "ScreenPorch", "PoolArea"
                          ,"WoodDeckSF","OpenPorchSF","EnclosedPorch", "LowQualFinSF"
                          ]
numerics = dataframe.loc[:,numericColumns]

# split into input (X) and output (Y) variables
y = dataframe.loc[:,'SalePrice']
x = numerics

# Do something for test set
x_sub = submission.loc[:,numericColumns]


## Do some preprocessing on the data
# Impute Missing Data
imp = pp.Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(x)
x_imputed = imp.transform(x)  
x_imp_sub = imp.transform(x_sub)  


# Scale
scaler = pp.StandardScaler().fit(x_imputed)
# scaler.scale_                                       
x_scaled = scaler.transform(x_imputed)  
x_scaled_sub = scaler.transform(x_imp_sub)  


# Normalize
normalizer = pp.Normalizer().fit(x_scaled)
x_normalized = normalizer.transform(x_scaled)  
x_normalized_sub = normalizer.transform(x_scaled_sub)  

# Polynomial features
#poly = PolynomialFeatures(2)
#poly.fit_transform(X) 

# Send data to splitter
featureMatrix = x_normalized
featureMatrix_sub = x_normalized_sub

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(featureMatrix, y, test_size=0.33, random_state=42)



## Lasso model
reg = linear_model.Lasso(alpha = 0.1)
# Fit model
reg.fit(X_train, y_train)

## Test model performance
prediction = reg.predict(X_test)

results= pd.DataFrame({'prediction':prediction, 'observation':y_test.astype('float')})

results.head(10)

rmse( results.prediction, results.observation )
#46089.00493046904

# Score some data for kaggle submission
prediction = reg.predict(featureMatrix_sub)

# Output Scores to CSV
pred_df = pd.DataFrame(prediction, index=submission["Id"], columns=["SalePrice"])
pred_df.to_csv('/Users/andrew/Documents/kaggle/realestate/python/output.csv', header=True, index_label='Id')

# Get current working dir
#import os
#cwd = os.getcwd()

# Make a keras model
def myModel():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=32, init='normal', activation='relu'))
	model.add(Dense(2, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


seed=5673
np.random.seed(seed)

model = KerasRegressor(build_fn=myModel, nb_epoch=200, batch_size=10, verbose=2)

## Dont know how to get the model out of this for scoring
# evaluate using 10-fold cross validation
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#results = cross_val_score(model, X_train, y_train, cv=kfold)
#print(results.mean())

# Fit model
model.fit(X_train, y_train)

# Predict model
prediction = model.predict(X_test)
pd.DataFrame(prediction, y_test).head(5)

res= pd.DataFrame({'prediction':prediction, 'observation':y_test.astype('float')})

rmse( res.prediction, res.observation )
# 68762.474618915265 worse than Lasso
# 50298.281266839957 MAC

# Score some data for kaggle submission
pred = model.predict(featureMatrix_sub)

# Output Scores to CSV
pred_df = pd.DataFrame(pred, index=submission["Id"], columns=["SalePrice"])
pred_df.to_csv('/Users/andrew/Documents/kaggle/realestate/python/output_NN.csv', header=True, index_label='Id')









 


# Output Scores to CSV
pred_df = pd.DataFrame(y_pred, index=test_df["Id"], columns=["SalePrice"])
pred_df.to_csv('/home/andrew/Documents/kaggle/realestate/output.csv', header=True, index_label='Id')









