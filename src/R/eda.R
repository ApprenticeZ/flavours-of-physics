# eda script for flavours of physics
setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')

source('./src/R/utils.R')

train <- read.csv('data/training.csv')

# target variable
y <- train$signal

# remove the last 5 features
train <- train[,2:46]
summary(train)

# feature normalization
ntrain <- apply(train, 2, featNorm)
summary(ntrain)

saveHist(ntrain,'./images')