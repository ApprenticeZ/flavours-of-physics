# eda script for flavours of physics
setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')

train <- read.csv('data/training.csv')

# target variable
y <- train$signal

train <- train[,-49]

