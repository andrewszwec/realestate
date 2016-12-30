###
###
### KAGGLE
### Leaf Classification
###
### Author: Andrew Szwec
### Date:   15/10/2016
###
### PLAN
### 1. Load data
### 2. Explore data
### 3. Prepare data
### 4. Train model
### 5. Test model performance using RMSE (Root mean square error)
### 6. Repeat
### 
### 

# Install this!
#install.packages("https://h2o-release.s3.amazonaws.com/h2o-ensemble/R/h2oEnsemble_0.1.8.tar.gz", repos = NULL)
#install.packages("RANN")
#install.packages("caret")

### 
### IMPORTS
### 
require(caret)
require(RANN)
require(h2oEnsemble)
require(mice)
require(VIM)


###
### Parameters
###
columnUniqueness = 0.15
runAll = 0 # reruns all data prep
splitRaio = 0.7

### 
### FUNCTIONs
### 
## evaluation metric
RMSE <- function (pred, obs){
  RMSE <- sqrt(mean(  (log(obs)-log(pred))^2 , na.rm=TRUE))
  
  return( RMSE )
}
'%!in%' <- function(x,y)!('%in%'(x,y))


### 
### LOAD DATA
### 
setwd("C:/Users/aszwec/Documents/kaggle 101/realestate")

df <- read.csv('train.csv')
sub <- read.csv('test.csv')

########################################################
## Missing Values in data
########################################################


# Plot Missing Values
if(runAll == 1){
  aggr_plot <- aggr(df, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, labels=names(df), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))
  
  ## Max allowable missing values is 6% Remove cols with > 6% missing
  
  a <- aggr_plot$missings[order(aggr_plot$missings$Count, decreasing = T),]
  a$percent <- a$Count / nrow(df)
  
  a[a$percent>0.06,]
  
  # Variables sorted by number of missings: 
  #   Variable        Count
  # PoolQC          0.9952054795
  # MiscFeature     0.9630136986
  # Alley           0.9376712329
  # Fence           0.8075342466
  # FireplaceQu     0.4726027397
  # LotFrontage     0.1773972603
  
  tooManyMissing <- a[a$percent>0.06,"Variable"]
  
  df <- subset(df, select = (names(df)[names(df) %!in% tooManyMissing]))
  
  # Impute Missing Values using Random Forest
  tempData <- mice(df,m=5,maxit=5,method='rf',seed=500)
  summary(tempData)
}

#save(tempData, file='Imputed training data.Rdata')
load('Imputed training data.Rdata')
df <- tempData$data

## Do the same to the submission data set
if(runAll == 1){
  sub <- subset(sub, select = (names(sub)[names(sub) %!in% tooManyMissing]))
  # Impute Missing Values
  tempData.sub <- mice(sub,m=5,maxit=5,method='rf',seed=500)
  summary(tempData.sub)
}
#save(tempData.sub, file='Imputed submission data.Rdata')
load('Imputed submission data.Rdata')
sub <- tempData.sub$data

########################################################
## DATA PREP - TRAINING - NUMERICS
########################################################

df$YearBuilt <- as.numeric(df$YearBuilt)
df$GarageYrBlt <- as.numeric(df$GarageYrBlt)
df$YrSold <- as.numeric(df$YrSold)
df$YearRemodAdd <- as.numeric(df$YearRemodAdd)

# Get all the numeric cols
xx <- subset(df, select=c('MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'MasVnrArea', 'BsmtFinSF1'
                          ,'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','X1stFlrSF','X2ndFlrSF'
                          ,'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'
                          ,'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea'
                          ,"GarageYrBlt", "YrSold", "YearBuilt",    "YearRemodAdd" , "MiscVal", "MoSold", "X3SsnPorch", "ScreenPorch", "PoolArea"
                          ,"WoodDeckSF","OpenPorchSF","EnclosedPorch", "LowQualFinSF"
                          ))

# Dont need anymore since we imputed above using MICE
#xx.p <- preProcess(xx, method = c('scale', 'center','knnImpute'))  # Impute missing values
#yy <- predict(xx.p, newdata=xx)
#yy <- data.frame(yy, SalePrice=df$SalePrice)
# sapply(yy, class)

yy <- data.frame(xx, SalePrice=df$SalePrice)

########################################################
## DATA PREP - SUBMISSION - NUMERICS
########################################################
sub$YearBuilt <- as.numeric(sub$YearBuilt)
sub$GarageYrBlt <- as.numeric(sub$GarageYrBlt)
sub$YrSold <- as.numeric(sub$YrSold)
sub$YearRemodAdd <- as.numeric(sub$YearRemodAdd)

zz <- subset(sub, select=c('MSSubClass', 'LotArea', 'OverallQual','OverallCond', 'MasVnrArea', 'BsmtFinSF1'
                           ,'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','X1stFlrSF','X2ndFlrSF'
                           ,'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'
                           ,'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea'
                           ,"GarageYrBlt", "YrSold", "YearBuilt",    "YearRemodAdd" , "MiscVal", "MoSold", "X3SsnPorch", "ScreenPorch", "PoolArea"
                           ,"WoodDeckSF","OpenPorchSF","EnclosedPorch", "LowQualFinSF"
))

# Dont need anymore since we imputed above using MICE
# zz.p <- preProcess(zz, method = c('scale', 'center','knnImpute')) ## NZV and Yeo Johnson Transform not helpful
# ss <- predict(zz.p, newdata=zz)

ss <- zz

########################################################
## DATA PREP - TRAINING - CATEGORICALS
########################################################
# separate out the categoricals
numeric.names <-c('SalePrice','MSSubClass','LotFrontage', 'LotArea', 'OverallQual','OverallCond', 'MasVnrArea', 'BsmtFinSF1'
                  ,'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','X1stFlrSF','X2ndFlrSF'
                  ,'GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr'
                  ,'KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea'
                  ,"GarageYrBlt", "YrSold", "YearBuilt",    "YearRemodAdd" , "MiscVal", "MoSold", "X3SsnPorch", "ScreenPorch", "PoolArea"
                  ,"WoodDeckSF","OpenPorchSF","EnclosedPorch", "LowQualFinSF"
)


categoricals <- setdiff(names(df), numeric.names)

# Make a dataframe of all categorical variables
train.cats <- subset(df, select = categoricals)

# Do near zero variance to remove columns with low information gain
nzz <- nzv(train.cats, saveMetrics = T, uniqueCut=columnUniqueness)
# Remove low information gain columns
train.cats <- train.cats[,!nzz$nzv]
dim(train.cats)

# do one-hot encoding of categoricals
dummies.tr <- dummyVars(Id ~ ., data = train.cats)
dummy.var.train <- predict(dummies.tr, newdata = train.cats)

########################################################
## DATA PREP - SUBMISSION - CATEGORICALS
########################################################

# Make a dataframe of all categorical variables
sub.cats <- subset(sub, select = categoricals)
# Remove NZV columns
sub.cats <- sub.cats[,!nzz$nzv]

# do one-hot encoding of categoricals
dummies.sb <- dummyVars(Id ~ ., data = sub.cats)
dummy.var.sub <- predict(dummies.sb, newdata = sub.cats)

########################################################
## DATA PREP - TRAINING - PUTTING IT ALL BACK TOGETHER
########################################################
# yy | dummy.var.train

yyy <- data.frame(yy, dummy.var.train)


########################################################
## DATA PREP - SUBMISSION - PUTTING IT ALL BACK TOGETHER
########################################################
# ss | dummy.var.sub

sss <- data.frame(ss, dummy.var.sub)

########################################################
## START H2O
########################################################
h2o.init(nthreads = -1)

### 
### RELOAD THE OLD MODEL FOR VIEWING
### 
#h2o.load_ensemble(path = "C:/Users/aszwec/Documents/kaggle 101/realestate/models/", import_levelone = TRUE)



# import data frame
df.hex <- as.h2o(yyy)


# Split data frame
df.split = h2o.splitFrame(data = df.hex, ratios = splitRaio)
df.train= df.split[[1]]
df.test = df.split[[2]]


## INITIALISE SOME LEARNERS
h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha)
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha)
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 50, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)


# Set up ensemble parameters
family <- "gaussian"
metalearner <- "h2o.glm.wrapper"

# learner <- c("h2o.glm.wrapper", "h2o.glm.3",
#              "h2o.randomForest.1", "h2o.randomForest.2", "h2o.randomForest.3", "h2o.randomForest.4",
#              "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8",
#              "h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7")

# learner <- c("h2o.glm.wrapper",
#              "h2o.randomForest.1", "h2o.randomForest.2",
#              "h2o.gbm.1", "h2o.gbm.6", "h2o.gbm.8",
#              "h2o.deeplearning.1", "h2o.deeplearning.6", "h2o.deeplearning.7") ## works best

learner <- c("h2o.glm.1", "h2o.glm.2", "h2o.glm.3",
             "h2o.randomForest.1", "h2o.randomForest.2", "h2o.randomForest.3", "h2o.randomForest.4",
             "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3", "h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6", "h2o.gbm.7","h2o.gbm.8"
             ) ## new 19/12/2016 12:57PM

# make x and y
y <- "SalePrice"
x <- setdiff(names(df.train), y)
# Build ensemble
fit <- h2o.ensemble(x = x, y = y, 
                    training_frame = df.train, 
                    family = family, 
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 20))

# Test model performance
perf <- h2o.ensemble_performance(fit, newdata = df.test)
perf
# 1054726535.71763
# 1040000670.03277
# 681923624.14596
# 843209059.950096
# 596947553.951224
# 684214383.257697

# impute, centre and scale
# impute, log, centre and scale

# Make predictions
pred <- predict(fit, newdata = df.test)

# Get predictions from h2o
predictions <- as.data.frame(pred$pred)  
obs <- as.data.frame(df.test[,y])[,1]

# Compare predictions to observations
cbind(predictions, obs)[1:20,]

RMSE(predictions, obs)
# 0.1013949 2016-12-19 12:11PM
# 0.1462425 2016-12-19 1:55PM

# make file name
#filename=gsub(':','',paste0("Ensemble Cats Numerics ",Sys.time(), ".Rdata"))

# make a new dir
mainDir = "C:/Users/aszwec/Documents/kaggle 101/realestate/models"
subDir = gsub(':','',paste0("Model ",Sys.time()))
path <- file.path(mainDir, subDir)
dir.create(path, showWarnings = FALSE)


# SAve Model
h2o.save_ensemble(fit, path = path, force = FALSE, export_levelone = TRUE)

# print Ensemble model
print(fit)


########################################################
## SCORES
########################################################
sub.hex <- as.h2o(sss)

pred.sub <- predict(fit, newdata = sub.hex)

# Get predictions from h2o
predictions.sub <- as.data.frame(pred.sub$pred)  

submission <- data.frame(Id=sub$Id, SalePrice=predictions.sub)
names(submission) <- c('Id', 'SalePrice')


# Make file name for submission
filename=gsub(':','',paste0("Resutls Ensemble ",Sys.time(), ".csv"))
write.csv(submission, file=filename, row.names = F)

h2o.shutdown()


########################################################
## NOW TRY XG BOOST
########################################################
require(xgboost)

########################################################
## SPLIT INTO TRAIN TEST AND VALIDATION
########################################################
# Split the data into a model training and test set used to measure the performance of the algorithm
set.seed(556677)
inTrain     = createDataPartition(yyy$SalePrice, p = 0.6)[[1]]
training    = yyy[ inTrain,]      # 60% of records
temp        = yyy[-inTrain,]      # 40% of reocrds

inTemp      = createDataPartition(temp$SalePrice, p = 0.5)[[1]]
validation  = temp[ inTemp,]      # 20% of records
testing     = temp[-inTemp,]      # 20% of reocrds


# Split the data into labels and features
Y <- training[, 'SalePrice']
X <- subset(training, select=-c(SalePrice))


## Find out what is the best number of iterations (50 is best)
mysequence <- seq(from = 10, to = 100, by =10)
rmse_results <- rep(0, length(mysequence))
j = 1

for(i in mysequence){
  # Train the model
  xgb <- xgboost(data = data.matrix(X), 
                 label = data.matrix(Y), 
                 eta = 0.1,
                 max_depth = 15, 
                 nround=i, 
                 subsample = 0.5,
                 colsample_bytree = 0.5,
                 seed = 1,
                 eval_metric = "rmse", # same evaluation metric as competition
                 objective = "reg:linear",  # Regression
                 nthread = 3
  )
  
  # Split test data into label and features
  Y_test <- testing[, 'SalePrice']
  X_test <- subset(testing, select=-c(SalePrice))
  
  y_pred <- predict(xgb, data.matrix(X_test))
  
  # Check model performance
  rmse_results[j] <- RMSE(y_pred, Y_test)
  #  0.1645949
  j = j + 1
  
}
# Plot the resutls to find the best number of rounds
plot(mysequence, rmse_results)


## Make model
xgb <- xgboost(data = data.matrix(X), 
               label = data.matrix(Y), 
               eta = 0.1,
               max_depth = 15, 
               nround=50, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "rmse", # same evaluation metric as competition
               objective = "reg:linear",  # Regression
               nthread = 3
)

# Split test data into label and features
Y_test <- testing[, 'SalePrice']
X_test <- subset(testing, select=-c(SalePrice))

y_pred <- predict(xgb, data.matrix(X_test))

# Check model performance
 RMSE(y_pred, Y_test)

# Score submission data :-)
y_pred <- predict(xgb, data.matrix(sss))
 

submission <- data.frame(Id=sub$Id, SalePrice=y_pred)
names(submission) <- c('Id', 'SalePrice')


# Make file name for submission
filename=gsub(':','',paste0("Results XGBOOST ",Sys.time(), ".csv"))
write.csv(submission, file=filename, row.names = F)
 
 
 
 
 
 