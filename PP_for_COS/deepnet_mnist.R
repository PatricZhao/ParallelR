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
