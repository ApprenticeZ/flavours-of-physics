library('ggplot2')

featNorm <- function(x){
    return ((x-mean(x))/sd(x))
}

featScale <- function(x){
  minx <- min(x)
  maxx <- max(x)
  return ((x-minx)/(maxx-minx)+minx)
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

saveScatter <- function(trainMat,target,path='./images/scatter'){
    nFeature <- dim(trainMat)[2]
    featName  <- colnames(trainMat)
    for (i in 1:nFeature){
        print(featName[i])
        filename <- paste0(path,'/',featName[i],'.png')
        dat <- data.frame(trainMat[,i],as.factor(target))
        colnames(dat) <- c(featName[i],'signal')
        p <- ggplot(dat, aes_string(x=featName[i], colour='signal')
                    ,xlab=featName[i],ylab='density'
                    ,main=paste(featName[i], '-signal scatter plot')) + geom_density()
        ggsave(file=filename,p)
        rm(p)
    }
}

saveBox <- function(trainMat,target,path='./images/box'){
    nFeature <- dim(trainMat)[2]
    featName  <- colnames(trainMat)
    for (i in 1:nFeature){
        print(featName[i])
        filename <- paste0(path,'/',featName[i],'.png')
        dat <- data.frame(trainMat[,i],as.factor(target))
        colnames(dat) <- c(featName[i],'signal')
        p <- ggplot(dat, aes_string(x='signal', y=featName[i], colour='signal')
                    ,xlab=featName[i],main=paste(featName[i], '-signal boxplot')) + geom_boxplot()
        ggsave(file=filename,p)
        rm(p)
    }
}