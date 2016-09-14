# Examples for the R and Parallel Computing blog in COS website （cos.name)
# Author: Peng Zhao, 8/30/2016

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
