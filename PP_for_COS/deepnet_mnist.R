# Examples for the R and Parallel Computing blog in COS website （cos.name)
# Author: Peng Zhao, 9/18/2016

# Example 1: Using Parallel Libraries
# using 'deepnet' and 'mnist' dataset for example
# see the performance w/ and w/o parallel BLAS backend

#install.packages("data.table")
#install.packages("deepnet")

library(data.table)
library(deepnet)


# download MNIST dataset in below links
# https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/train.csv.gz
# https://h2o-public-test-data.s3.amazonaws.com/bigdata/laptop/mnist/test.csv.gz
mnist.train <- as.matrix(fread("./train.csv", header=F))
mnist.test  <- as.matrix(fread("./test.csv", header=F))

# V785 is the label
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





