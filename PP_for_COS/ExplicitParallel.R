# Examples for the R and Parallel Computing blog in COS website （cos.name)
# Author: Peng Zhao, 8/30/2016

# Toy case: solve quad equation  
len <- 1e8
a <- runif(len, -10, 10)
a[sample(len, 100,replace=TRUE)] <- 0

b <- runif(len, -10, 10)
c <- runif(len, -10, 10)

# Not vectorized function
solve.quad.eq <- function(a, b, c) 
{
  # Not validate eqution: a and c are almost ZERO
  if(abs(a) < 1e-8 && abs(b) < 1e-8) return(c(NA, NA) )
  
  # Not quad equation
  if(abs(a) < 1e-8 && abs(b) > 1e-8) return(c(-c/b, NA))
  
  # No Solution
  if(b*b - 4*a*c < 0) return(c(NA,NA))
  
  # Return solutions
  x.delta <- sqrt(b*b - 4*a*c)
  x1 <- (-b + x.delta)/(2*a)
  x2 <- (-b - x.delta)/(2*a)
  
  return(c(x1, x2))
}

#############################################################################################
# *apple style
##############################################################################################
# serial code
system.time(
    res1.s <- sapply(1:len, FUN = function(x) { solve.quad.eq(a[x], b[x], c[x])}, simplify = F)
)

# parallel
library(parallel)
# multicores on Linux
system.time(
  res1.p <- mclapply(1:len, FUN = function(x) { solve.quad.eq(a[x], b[x], c[x])}, mc.cores = 4)
)


# cluster on Windows
cores <- detectCores(logical = FALSE)
cl <- makeCluster(cores)
clusterExport(cl, c('solve.quad.eq', 'a', 'b', 'c'))
system.time(
   res1.p <- parLapply(cl, 1:len, function(x) { solve.quad.eq(a[x], b[x], c[x]) })
)
stopCluster(cl)




##########################################################################################
# For style
###########################################################################################
# serial code
res2.s <- matrix(0, nrow=len, ncol = 2)
system.time(
    for(i in 1:len) {
        res2.s[i,] <- solve.quad.eq(a[i], b[i], c[i])
    }
)

# foreach
library(foreach)
library(doParallel)

# Real physical cores in my computer
cores <- detectCores(logical = FALSE)
cl <- makeCluster(cores)
registerDoParallel(cl, cores=cores)

# clusterSplit are very convience to split data but it takes lots of extra memory
# chunks <- clusterSplit(cl, 1:len)

# split data by ourselves
chunk.size <- len/cores

system.time(
  res2.p <- foreach(i=1:cores, .combine='rbind') %dopar%
  {  # local data for results
     res <- matrix(0, nrow=chunk.size, ncol=2)
     for(x in ((i-1)*chunk.size+1):(i*chunk.size)) {
        res[x - (i-1)*chunk.size,] <- solve.quad.eq(a[x], b[x], c[x])
     }
     # return local results
     res
  }
)

stopImplicitCluster()
stopCluster(cl)


# BAD but direct implementation
# Real physical cores in my computer
# cores <- detectCores(logical=F)
# cl <- makeCluster(cores)
# registerDoParallel(cl, cores=cores)

#system.time(
#  res2.p <- foreach(i=1:len, .combine='rbind') %dopar%
#  {  
#      solve.quad.eq(a[i], b[i], c[i])
#  }
#)
#stopImplicitCluster()
#stopCluster(cl)


###############################################################################
# Results on Intel Xeon  (Linux System)
###############################################################################
> Rscript ExplicitParallel.R
   user  system elapsed
 78.914   0.445  79.435
   user  system elapsed
 21.644   1.485  23.153
[1] TRUE
   user  system elapsed
  4.448   0.244  23.868
[1] TRUE
   user  system elapsed
 53.081   0.071  53.202
Loading required package: iterators
   user  system elapsed
  1.460   0.168  15.327
[1] TRUE

