setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')
library('randomForest')
library('xgboost')

# load training/test data
train <- read.csv('training.csv')
test <- read.csv('test.csv')

# remove `SPDhits` in train
vtrain <- train[,-47]

# train a random forest model
set.seed(1)
rf <- randomForest(as.factor(signal)~., data=vtrain, ntree=100)

# train a xgboost model
param_list <- list(max.depth=5
                              ,eta=0.3
                              ,objective="binary:logistic"
                              ,min_child_weight=3
                              ,silent=1
                              ,subsample=0.7
                              ,colsample_bytree=0.7
                              ,seed=1)
dtrain <- xgb.DMatrix(data.frame(vtrain[,-48]),vtrain$signal)
bst <- xgb.train(param_list,dtrain,nround=250)

# make predictions on test