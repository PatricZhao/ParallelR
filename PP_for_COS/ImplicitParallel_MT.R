# Examples for the R and Parallel Computing blog in COS website （cos.name)
# Author: Peng Zhao, 8/30/2016

# comparison of single thread and multiple threads run
# using Internal function to set thread numbers, not very grace, but don't find a good way till now
# suggestions?
for(i in 6:11) {
    ORDER <- 2^i
    m <- matrix(rnorm(ORDER*ORDER),ORDER,ORDER);
    .Internal(setMaxNumMathThreads(1)); .Internal(setNumMathThreads(1)); res <- system.time(d <- dist(m))
     print(res)
    .Internal(setMaxNumMathThreads(20)); .Internal(setNumMathThreads(20)); res <- system.time(d <- dist(m))
     print(res)
}
