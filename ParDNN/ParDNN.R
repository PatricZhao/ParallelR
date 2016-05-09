#   Copyright: GTC2016, Unblock Deep Neural Network Performance Limiter by CUDA.
#   Auther: Patric Zhao, Nvidia,


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
  # init loss to a very big value
  loss <- 100000
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
  # init loss to a very big value
  loss <- 100000
  
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
  # init loss to a very big value
  loss <- 100000
  
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


# Train:  Optimization 2: remove sweep() functions
#                         combine W and b together          
train.dnn.cuda <- function(x, y, 
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
  # init loss to a very big value
  loss <- 100000
  
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
    hidden.layer <- pmax.cuda(hidden.layer, 0)
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


# Optimization 4: implement GEMM by CDUA
train.dnn.cublas <- function(x, y, 
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
                         random.seed = 1,
                         # can be multiple c(0,1,2,3)
                         devID = 0)
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
  # init loss to a very big value
  loss <- 100000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol ) {
    
    # iteration index
    i <- i +1
    
    # forward ....
    # Opt2: remove `sweep` 
    #hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    #hidden.layer <- X1 %*% W1b1
    hidden.layer <- cuBLAS(X1 , W1b1, devID=devID)
    # neurons : ReLU
    hidden.layer <- pmax.cuda(hidden.layer, 0, devID=devID)
    # Opt2: remove `sweep`
    #score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    hidden.layer1 <- cbind(hidden.layer, rep(1,nrow(hidden.layer)))
    #score <- hidden.layer1 %*% W2b2
    score <- cuBLAS(hidden.layer1, W2b2, devID=devID)
    
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
    # dW2 <- crossprod(hidden.layer,dscores)
    # Opt4: call cuda API directly
    dW2 <- cuBLAS(hidden.layer, dscores, transA=T, devID=devID)
    db2 <- colSums(dscores)
    
    # Opt1: replace t() operations
    #dhidden <- tcrossprod(dscores, W2)
    # Opt4: call cuda API directly
    dhidden <- cuBLAS(dscores, W2, transB=T, devID=devID)
    #dhidden[hidden.layer <= 0] <- 0
    
    # Opt1: replace t() operations
    # dW1 <- crossprod(X, dhidden)
    # Opt4: call cuda API directly
    dW1 <- cuBLAS(X, dhidden, transA=T, devID=devID)
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



# Parallel model for multi-GPU and CPU
train.dnn.para <-  function(x, y, traindata=data, testdata=NULL,
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
                            random.seed = 1,
                            # specify CPU, GPU, MIC 
                            devType = c("GPU"),
                            # number of parallel devices
                            devNum = 4, 
                            # iteration in each devices
                            devIter = 10) 
{
  #library("parallel") 
  # data decomposition
  N <- nrow(traindata)
  N.examples <- as.integer(N/devNum)
  
  # data decoposition
  N.para  <- c(rep(N.examples, devNum-1), N.examples+ N %% N.examples)
  N.start <- cumsum(c(1,N.para))[1:devNum]
  N.end   <- cumsum(N.para)
  
  # init model : only init the parameter w/o real training
  para.model <- train.dnn.O2(x, 
                          y, 
                          traindata=traindata[N.start[1]:N.end[1],], 
                          testdata=testdata, 
                          hidden=hidden, 
                          maxit=0,
                          lr=lr,
                          reg=reg,
                          random.seed=random.seed) 
  
  
  # init variable
  ii  <- 0
  loss <- 1e5
  while (ii < maxit && loss > abstol) {
    
    if(devType == "GPU") {
       res <- mclapply(1:devNum, function(id) { 
                                              train.dnn.cublas(x, 
                                              y, 
			                      omodel=para.model,
                                              traindata=traindata[N.start[id]:N.end[id],], 
                                              testdata=testdata, 
                                              hidden=hidden, 
                                              maxit=devIter,
                                              lr=lr,
                                              reg=reg,
                                              random.seed=random.seed,
                                              display=display,
			                      devID=(id-1)) 
                                 }, 
                     mc.cores=devNum, mc.preschedule=TRUE)
    } else {
        res <- mclapply(1:devNum, function(id) { 
                                               train.dnn.O2(x, 
                                               y, 
                                               omodel=para.model,
                                               traindata=traindata[N.start[id]:N.end[id],], 
                                               testdata=testdata, 
                                               hidden=hidden, 
                                               maxit=devIter,
                                               lr=lr,
                                               reg=reg,
                                               random.seed=random.seed,
                                               display=display) 
                                  }, 
                       mc.cores=devNum, mc.preschedule=TRUE)
    } #else

    # construct new model with parallel weights
    D <- res[[1]]$D
    H <- res[[1]]$H
    K <- res[[1]]$K
    if(devNum > 1) {
        for(i in 2:devNum) {
            res[[1]]$W1 <- res[[1]]$W1 + res[[i]]$W1
            res[[1]]$W2 <- res[[1]]$W2 + res[[i]]$W2
            res[[1]]$b1 <- res[[1]]$b1 + res[[i]]$b1
            res[[1]]$b2 <- res[[1]]$b2 + res[[i]]$b2
        }
    }
    
    para.model <- list( D = D,
                        H = H,
                        K = K,
                        # weights and bias
                        W1= res[[1]]$W1/devNum, 
                        b1= res[[1]]$b1/devNum, 
                        W2= res[[1]]$W2/devNum, 
                        b2= res[[1]]$b2/devNum)
    
    ii <- ii +1
  }
  
  return(para.model)
}

