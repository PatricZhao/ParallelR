# ParallelR.com : CUDA and multiGPU
# Author: Peng Zhao

# HoGWILD
# 1. source our function
library(parallel)
source("ParDNN.R")
source("cudaR.R")

# 2. setup date
train <- read.csv('../data/train.csv', header=F)
test  <- read.csv('../data/test.csv', header=F)


train <- data.matrix(train)
test <- data.matrix(test)

# normlization
train[,1:(ncol(train)-1)] <- train[,1:(ncol(train)-1)]/255
test[,1:(ncol(test)-1)]   <- test[,1:(ncol(test)-1)]/255

# performance testing for different number of devices(CPU/GPU)
for(i in c(8,6,4,2, 1)) {
time.dnn <- system.time(
para.model <- train.dnn.para(x=1:784,
                             y=785,
                             traindata=train,
                             hidden=64,
                             maxit=1,
                             display=100,
                             lr=0.1,
                             reg=1e-3,
                             devType="GPU",
                             devNum=i,
                             devIter=200)
)
print(time.dnn)
}


