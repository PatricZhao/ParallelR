# copyright 2016: www.ParallelR.com
# Author: Peng Zhao, patric.zhao@gmail.com
# R For Deep Learning (II): Achieve High-Performance DNN With Parallel Acceleration 

# Prediction
predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
  # neurons : Rectified Linear
  hidden.layer <- pmax(hidden.layer, 0)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  score.exp <- exp(score)
  probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs)
  return(labels.predicted)
}

# Train: build and train a 2-layers neural network 
train.dnn <- function(x, y, traindata=data, testdata=NULL,
                  # set hidden layers and neurons
                  # currently, only support 1 hidden layer
                  hidden=c(6), 
                  # max iteration steps
                  maxit=2000,
                  # delta loss 
                  abstol=1e-2,
                  # learning rate
                  lr = 1e-2,
                  # regularization rate
                  reg = 1e-3,
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
  if(is.factor(Y)) { Y <- as.integer(Y) }
  # create index for both row and col
  Y.len   <- length(unique(Y))
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  # number of input features
  D <- ncol(X)
  # number of categories for classification
  K <- Y.len
  H <- hidden
  
  # create and init weights and bias 
  W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
  b1 <- matrix(0, nrow=1, ncol=H)
  
  W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
  b2 <- matrix(0, nrow=1, ncol=K)
 
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # March 18, 2016, updated: init with a big number
  loss <- 10000
  
  # Training the network
  i <- -1
  while(i < maxit && loss > abstol ) {
    # iteration index
    i <- i +1
    
    # forward ....
    # 1 indicate row, 2 indicate col
    hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    # neurons : ReLU
    hidden.layer <- pmax(hidden.layer, 0)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    
    # softmax
    score.exp <- exp(score)
    probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
    
    # compute the loss
    corect.logprobs <- -log(probs[Y.index])
    data.loss  <- sum(corect.logprobs)/batchsize
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
            labs <- predict.dnn(model, testdata[,-y])
            accuracy <- mean(as.integer(testdata[,y]) == Y.set[labs])
            cat(i, loss, accuracy, "\n")
        } else {
            cat(i, loss, "\n")
        }
    }
    
    # backward ....
    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] -1
    dscores <- dscores / batchsize
    
    
    dW2 <- t(hidden.layer) %*% dscores 
    db2 <- colSums(dscores)
    
    dhidden <- dscores %*% t(W2)
    dhidden[hidden.layer <= 0] <- 0
    
    dW1 <- t(X) %*% dhidden
    db1 <- colSums(dhidden) 
    
    # update ....
    dW2 <- dW2 + reg*W2
    dW1 <- dW1  + reg*W1
    
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


# Train:  Optimization 1: remove t() functions
#                      2: add model input to retrain                    
train.dnn.O1 <- function(x, y, 
                         traindata=data, 
                         testdata=NULL,
                         omodel=NULL,
                  # set hidden layers and neurons
                  # currently, only support 1 hidden layer
                  hidden=c(6), 
                  # max iteration steps
                  maxit=2000,
                  # delta loss 
                  abstol=1e-2,
                  # learning rate
                  lr = 1e-2,
                  # regularization rate
                  reg = 1e-3,
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
  if(is.factor(Y)) { Y <- as.integer(Y) }
  # create index for both row and col
  Y.len   <- length(unique(Y))
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  
  # create model or get model from parameter
  if(is.null(omodel)) {
    # number of input features
    D <- ncol(X)
    # number of categories for classification
    K <- length(unique(Y))
    H <-  hidden
    
    # create and init weights and bias 
    W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
    b1 <- matrix(0, nrow=1, ncol=H)
    
    W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
    b2 <- matrix(0, nrow=1, ncol=K)
  } else {
    D  <- omodel$D
    K  <- omodel$K
    H  <- omodel$H
    W1 <- omodel$W1
    b1 <- omodel$b1
    W2 <- omodel$W2
    b2 <- omodel$b2
  }
    
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # March 18, 2016, updated: init with a big number
  loss <- 10000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol ) {
    
    # iteration index
    i <- i +1
    
    # forward ....
    # 1 indicate row, 2 indicate col
    hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    # neurons : ReLU
    hidden.layer <- pmax(hidden.layer, 0)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    
    # softmax
    score.exp <- exp(score)
    # ?? further optimization in here
    probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
    
    # compute the loss
    corect.logprobs <- -log(probs[Y.index])
    data.loss  <- sum(corect.logprobs)/batchsize
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
            labs <- predict.dnn(model, testdata[,-y])
            accuracy <- mean(as.integer(testdata[,y]) == Y.set[labs])
            cat(i, loss, accuracy, "\n")
        } else {
            cat(i, loss, "\n")
        }
    }
    
    # backward ....
    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] -1
    dscores <- dscores / batchsize
    
    # Optimization for data transform
    # dW2 <- t(hidden.layer) %*% dscores 
    dW2 <- crossprod(hidden.layer, dscores)
    db2 <- colSums(dscores)
    

    # Optimization for data transform
    # dhidden <- dscores %*% t(W2)
    dhidden <- tcrossprod(dscores, W2)
    dhidden[hidden.layer <= 0] <- 0
    
    # Optimization for data transform
    # dW1 <- t(X) %*% dhidden
    dW1 <- crossprod(X, dhidden)
    db1 <- colSums(dhidden) 
    
    # update ....
    dW2 <- dW2 + reg*W2
    dW1 <- dW1  + reg*W1
    
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

# Train:  Optimization 2: remove sweep() functions
#                         combine W and b together          
train.dnn.O2 <- function(x, y, 
                         traindata=data, 
                         testdata=NULL,
                         omodel=NULL,
                         # set hidden layers and neurons
                         # currently, only support 1 hidden layer
                         hidden=c(6), 
                         # max iteration steps
                         maxit=2000,
                         # delta loss 
                         abstol=1e-2,
                         # learning rate
                         lr = 1e-2,
                         # regularization rate
                         reg = 1e-3,
                         # show results every 'display' step
                         display = 100,
                         random.seed = 1)
{
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- nrow(traindata)
  
  # extract the data and label
  X <- data.matrix(unname(traindata[,x]))
  
  # correct categories represented by integer 
  Y <- traindata[,y]
  if(is.factor(Y)) { Y <- as.integer(Y) }
  # create index for both row and col
  Y.len   <- length(unique(Y))
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  # create model or get model from parameter
  if(is.null(omodel)) {
    # number of input features
    D <- ncol(X)
    # number of categories for classification
    K <- length(unique(Y))
    H <-  hidden
    
    # create and init weights and bias 
    W1   <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
    b1   <- matrix(0, nrow=1, ncol=H)
    
    
    W2   <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
    b2   <- matrix(0, nrow=1, ncol=K)
    
  } else {
    D  <- omodel$D
    K  <- omodel$K
    H  <- omodel$H
    W1 <- omodel$W1
    b1 <- omodel$b1
    W2 <- omodel$W2
    b2 <- omodel$b2
  }
  
  # Opt2: combine data and add 1 column for bias
  # backsize, waste more memory!
  W1b1 <- rbind(W1, b1)
  W2b2 <- rbind(W2, b2)
  X1   <- cbind(X, rep(1, nrow(X)))
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # March 18, 2016, updated: init with a big number
  loss <- 10000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol ) {
    
    # iteration index
    i <- i +1
    
    # forward ....
    # Opt2: remove `sweep` 
    #hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    hidden.layer <- X1 %*% W1b1
    # neurons : ReLU
    hidden.layer <- pmax(hidden.layer, 0)
    # Opt2: remove `sweep`
    #score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    hidden.layer1 <- cbind(hidden.layer, rep(1,nrow(hidden.layer)))
    score <- hidden.layer1 %*% W2b2
    
    # softmax
    score.exp <- exp(score)
    probs <-score.exp/rowSums(score.exp) 
    
    # compute the loss
    corect.logprobs <- -log(probs[Y.index])
    data.loss  <- sum(corect.logprobs)/batchsize
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
        labs <- predict.dnn(model, testdata[,-y])
        accuracy <- mean(as.integer(testdata[,y]) == Y.set[labs])
        cat(i, loss, accuracy, "\n")
      } else {
        cat(i, loss, "\n")
      }
    }
    
    # backward ....
    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] -1
    dscores <- dscores / batchsize
    
    # Opt1: replace t() operations
    dW2 <- crossprod(hidden.layer,dscores)
    db2 <- colSums(dscores)
    
    # Opt1: replace t() operations
    dhidden <- tcrossprod(dscores, W2)
    dhidden[hidden.layer <= 0] <- 0
    
    # Opt1: replace t() operations
    dW1 <- crossprod(X, dhidden)
    db1 <- colSums(dhidden) 
    
    # update ....
    dW2 <- dW2  + reg*W2
    dW1 <- dW1  + reg*W1
    
    W1 <- W1 - lr * dW1
    b1 <- b1 - lr * db1
    
    W2 <- W2 - lr * dW2
    b2 <- b2 - lr * db2
    
    # Opt2: update combinations
    W1b1 <- rbind(W1, b1)
    W2b2 <- rbind(W2, b2)
  
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

