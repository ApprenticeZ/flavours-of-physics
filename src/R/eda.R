# eda script for flavours of physics
setwd('/media/zhang/01CE0A9C75603BA0/pg-course/kdd/tau-mu/')

source('./src/R/utils.R')

train <- read.csv('data/training.csv')
test <- read.csv('data/test.csv')

# feature normalization
ntrain <- apply(train[,c(-1,-49)], 2, featNorm)
dtrain <- data.frame(train[,c(1,49)],ntrain)
rm(ntrain)
colnames(dtrain)[1:2] <- c('id','signal')

strain <- apply(train[,c(-1,-49)], 2, featScale)
sdtrain <- data.frame(train[,c(1,49)],strain)
rm(strain)
colnames(sdtrain)[1:2] <- c('id','signal')

ntest <- apply(test[,-1],2,featNorm)
dtest <- data.frame(test[,1],ntest)
rm(ntest)
colnames(dtest)[1] <- 'id'
write.csv(dtest, './data/ntest.csv', quote=FALSE, row.names=FALSE)

stest <- apply(test[,-1],2,featScale)
sdtest <- data.frame(test[,1],stest)
rm(stest)
colnames(sdtest)[1] <- 'id'
write.csv(sdtest, './data/stest.csv', quote=FALSE, row.names=FALSE)


saveHist(dtrain[,2:51],'./images/hist')
saveHist(train[,2:51],'./images/raw-hist')

saveScatter(train[,c(-1,-49)],train$signal,'./images/raw-scatter')
saveScatter(sdtrain[,c(-1,-2)],sdtrain$signal,'./images/scale-hist')

saveBox(train[,c(-1,-49)],train$signal,'./images/raw-box')
saveBox(dtrain[,c(-1,-2)],dtrain$signal,'./images/box')
saveBox(sdtrain[,c(-1,-2)],sdtrain$signal,'./images/scale-box')
