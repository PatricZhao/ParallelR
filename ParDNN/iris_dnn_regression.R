# Copyright 2016: www.ParallelR.com
# Parallel Blog : R For Deep Learning (I): Build Fully Connected Neural Network From Scratch
# Regression by 2-layers DNN and tested by iris dataset
# Description: Build 2-layers DNN to predict Petal.Width according to the other three variables in iris dataset.
# Author: Peng Zhao, patric.zhao@gmail.com

# sigmoid
sigmoid <- function(z) {
  g <- 1 / (1 + exp(-1 * z))
  g
}

# Prediction
predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
  hidden.layer <- sigmoid(hidden.layer)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  Petal.Width.predicted <- score
  
  return(Petal.Width.predicted)
}

# Train: build and train a 2-layers neural network 
train.dnn <- function(x, y, traindata=data, testdata=NULL,
                      model = NULL,
                      # set hidden layers and neurons
                      # currently, only support 1 hidden layer
                      hidden = c(6), 
                      # max iteration steps
                      maxit = 2000,
                      # delta loss 
                      abstol = 1e-2,
                      # learning rate
                      lr = 1e-2,
                      # regularization rate
                      reg = 1e-4,
                      # show results every 'display' step
                      display = 100,
                      random.seed = 1)
{
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- nrow(traindata)
  
  # extract the data and label
  # don't need atribute 
  X <- unname(data.matrix(traindata[,x]))
  # correct categories represented by integer 
  Y <- traindata[,y]
  
  # create model or get model from parameter
  if(is.null(model)) {
    # number of input features
    D <- ncol(X)
    # only one output node for regression
    K <- 1
    H <-  hidden
    
    # create and init weights and bias 
    W1 <- matrix(rnorm(D*H), nrow=D, ncol=H)/sqrt(D*H)
    b1 <- matrix(0, nrow=1, ncol=H)
    
    W2 <- matrix(rnorm(H*K), nrow=H, ncol=K)/sqrt(H*K)
    b2 <- matrix(0, nrow=1, ncol=K)
  } else {
    D  <- model$D
    K  <- model$K
    H  <- model$H
    W1 <- model$W1
    b1 <- model$b1
    W2 <- model$W2
    b2 <- model$b2
  }
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # init loss to a very big value
  loss <- 100000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol) {
    
    # iteration index
    i <- i +1
    
    # forward ....
    # 1 indicate row, 2 indicate col
    hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    # neurons : sigmoid
    hidden.layer <- sigmoid(hidden.layer)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    Petal.Width <- score
    
    # compute the loss
    dif <- Petal.Width - Y
    data.loss <- (t(dif) %*% dif) / (2 * N)
    reg.loss   <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
    loss <- data.loss + reg.loss
    
    # display results and update model
    if( i %% display == 0) {
      if(!is.null(testdata)) {
        model <- list( D = D,
                       H = H,
                       K = K,
                       # weights and bias
                       W1 = W1, 
                       b1 = b1, 
                       W2 = W2, 
                       b2 = b2)
        Petal.Width <- predict.dnn(model, testdata[,-y])
        dif <- Petal.Width - testdata[,y]
        mse <- (t(dif) %*% dif) / (2 * N)
        cat(i, loss, mse, "\n")
      } else {
        cat(i, loss, "\n")
      }
    }
    
    # backward ....
    dscores <- Petal.Width - Y
    dscores <- dscores / batchsize
    
    dW2 <- t(hidden.layer) %*% dscores 
    db2 <- colSums(dscores)
    
    dhidden <- dscores %*% t(W2)
    
    dW1 <- dhidden * (hidden.layer * (1-hidden.layer))
    dW1 <- t(X) %*% dW1
    
    db1 <- dhidden * (hidden.layer * (1-hidden.layer))
    db1 <- colSums(db1)
    
    # update ....
    dW2 <- dW2 + reg*W2
    dW1 <- dW1 + reg*W1
    
    W1 <- W1 - lr * dW1
    b1 <- b1 - lr * db1
    
    W2 <- W2 - lr * dW2
    b2 <- b2 - lr * db2
    
    
    
  }
  
  # final results
  # creat list to store learned parameters
  # you can add more parameters for debug and visualization
  # such as residuals, fitted.values ...
  model <- list( D = D,
                 H = H,
                 K = K,
                 # weights and bias
                 W1= W1, 
                 b1= b1, 
                 W2= W2, 
                 b2= b2)
  
  return(model)
}

########################################################################
# testing
#######################################################################
set.seed(1)

# 0. split data into test/train
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))

# 1. EDA
data <- iris
data <- iris[,-5]
summary(data)
# 2. train model
Petal.Width.model <- train.dnn(x=1:3, y=4, traindata=data[samp,], testdata=data[-samp,], hidden=4, maxit=2000, display=50)
# ir.model <- train.dnn(x=1:3, y=4, traindata=iris[samp,], hidden=4, maxit=2000, display=50)

# 3. prediction
# To make the code clear, I don't write this change into predict.dnn function.
Petal.Width.dnn <- predict.dnn(Petal.Width.model, data[-samp, -4])

#mse
test <- data[-samp,]
m <- length(test[,4])
dif <- Petal.Width.dnn - test[,4]
mse <- (t(dif) %*% dif) / (2 * m)
mse #0.02664111

# Visualization
# the output from screen, copy and paste here.
data1 <- ("i loss mse
          50 0.1815593 0.1955971 
100 0.1543977 0.1645302 
          150 0.1359293 0.1447406 
          200 0.1193753 0.1273414 
          250 0.1043159 0.1116563 
          300 0.09069428 0.09754797 
          350 0.07858586 0.08504139 
          400 0.06807212 0.07418793 
          450 0.05917428 0.0649938 
          500 0.05183076 0.05739113 
          550 0.04590674 0.05124195 
          600 0.04122019 0.04636106 
          650 0.03757053 0.04254354 
          700 0.03476154 0.03958798 
          750 0.03261592 0.03731178 
          800 0.03098245 0.03555882 
          850 0.02973758 0.03420151 
          900 0.02878368 0.03313912 
          950 0.02804566 0.03229441 
          1000 0.02746694 0.03160948 
          1050 0.02700568 0.03104186 
          1100 0.02663133 0.03056087 
          1150 0.02632181 0.03014469 
          1200 0.02606127 0.02977792 
          1250 0.02583831 0.02944974 
          1300 0.02564474 0.02915253 
          1350 0.02547455 0.02888083 
          1400 0.02532334 0.02863066 
          1450 0.02518775 0.02839903 
          1500 0.02506522 0.0281836 
          1550 0.02495374 0.02798249 
          1600 0.02485169 0.02779412 
          1650 0.02475776 0.02761717 
          1700 0.02467087 0.02745047 
          1750 0.02459011 0.02729302 
          1800 0.02451474 0.02714392 
          1850 0.02444412 0.02700239 
          1900 0.02437769 0.02686773 
          1950 0.024315 0.02673932 
          2000 0.02425565 0.0266166 ")

data.v <- read.table(text=data1, header=T)
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(x=data.v$i, y=10*(data.v$loss), type="o", col="blue", pch=16, 
     main="IRIS loss and mse by 2-layers DNN",
     ylim=c(0, 2.0),
     xlab="",
     ylab="",
     axe =F)
lines(x=data.v$i, y=10*data.v$mse, type="o", col="red", pch=1)
box()
axis(1, at=seq(0,2000,by=200))
axis(4, at=seq(0.2,2.0,by=0.1))
axis(2, at=seq(0.2,2.0,by=0.1))
mtext("training step", 1, line=3)
mtext("loss of training set (*10)", 2, line=2.5)
mtext("mse of testing set (*10)", 4, line=2)

legend("bottomleft", 
       legend = c("loss", "mse"),
       pch = c(16,1),
       col = c("blue","red"),
       lwd=c(1,1)
)