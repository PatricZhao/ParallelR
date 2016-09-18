# Examples for the R and Parallel Computing blog in COS website （cos.name)
# Author: Peng Zhao, 8/30/2016

# Example 1: Using Parallel Libraries
# using 'deepnet' and 'mnist' dataset for example
# see the performance w/ and w/o parallel BLAS backend

#install.packages("data.table")
#install.packages("deepnet")

library(data.table)
library(deepnet)

# V785 is the label
mnist.train <- as.matrix(fread("./train.csv", header=F))
mnist.test  <- as.matrix(fread("./test.csv", header=F))

x <- mnist.train[, 1:784]/255
y <- model.matrix(~as.factor(mnist.train[, 785])-1)

system.time(
nn <- dbn.dnn.train(x,y,
                    hidden=c(64),
                    output="softmax",
                    batchsize=128, numepochs=100, learningrate = 0.1)
)



# OpenBLAS
# env LD_PRELOAD=/.../tools/OpenBLAS/lib/libopenblas.so R CMD BATCH deepnet_mnist.R

#begin to train dbn ......
#training layer 1 rbm ...
#dbn has been trained.
#begin to train deep nn ......
####loss on step 10000 is : 0.193343
####loss on step 20000 is : 0.121218
####loss on step 30000 is : 0.127029
####loss on step 40000 is : 0.159519
#deep nn has been trained.
#     user    system   elapsed 
# 2197.394 10496.190   867.748

# native R
# R CMD BATCH deepnet_mnist.R
#begin to train dbn ......
#training layer 1 rbm ...
#dbn has been trained.
#begin to train deep nn ......
####loss on step 10000 is : 0.179346
####loss on step 20000 is : 0.123266
####loss on step 30000 is : 0.136734
####loss on step 40000 is : 0.085222
#deep nn has been trained.
#    user   system  elapsed 
#2110.710    2.311 2115.042



# Example 2: Using MultiThreading Functions
# comparison of single thread and multiple threads run
# using Internal function to set thread numbers, not very grace, but don't find a good way till now.
# Ang suggestion?
setNumThreads <- function(nums=1) {
  .Internal(setMaxNumMathThreads(nums));
  .Internal(setNumMathThreads(nums));
}

# Testing dist funciton with single (1) and multiple (20) threads
for(i in 6:11) {
    ORDER <- 2^i
    m <- matrix(rnorm(ORDER*ORDER),ORDER,ORDER);
    setNumThreads(1)
    res <- system.time(d <- dist(m))
    print(res)
    setNumThreads(20)
    res <- system.time(d <- dist(m))
    print(res)
}

# Results
#user  system elapsed 
#  0.002   0.000   0.002 
#   user  system elapsed 
#  0.076   0.001   0.005 
#   user  system elapsed 
#  0.012   0.000   0.012 
#   user  system elapsed 
#  0.076   0.000   0.005 
#   user  system elapsed 
#  0.089   0.000   0.089 
#   user  system elapsed 
#  0.164   0.001   0.020 
#   user  system elapsed 
#  0.716   0.000   0.717 
#   user  system elapsed 
#  0.993   0.001   0.117 
#   user  system elapsed 
#  4.651   0.000   4.656 
#   user  system elapsed 
#  5.728   0.001   0.564 
#   user  system elapsed 
#136.895   0.005 137.030 
#   user  system elapsed 
#215.808   0.048  21.490 
