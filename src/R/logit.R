# eda script for flavours of physics
setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')

train <- read.csv('data/training.csv')
test <- read.csv('data/test.csv')

# target variable
y <- train$signal

train <- train[,1:46]