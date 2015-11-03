# eda script for flavours of physics
setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')

train <- read.csv('data/training.csv')
test <- read.csv('data/test.csv')

# target variable
y <- train$signal

# remove the unseen features in testset
train <- train[,2:46]

# feature normalization
featNorm <- function (x){
    return ((x-mean(x))/std(x))
}

train <- apply(train, featNorm, )
# fit a logistic regression model
fit <- glm(y~., data=train, family='gaussian')
summary(fit)
anova(fit)