featNorm <- function(x){
    return ((x-mean(x))/sd(x))
}

saveHist <- function(trainMat,path){
    nFeature <- dim(trainMat)[2]
    featName  <- colnames(trainMat)
    for (i in 1:nFeature){
        png(paste0(path,'/',featName[i],'.png'))
        hist(trainMat[,i],breaks=20,freq=FALSE
                ,xlab=featName[i],main=paste(featName[i], ' histogram'))
        dev.off()
    }
}