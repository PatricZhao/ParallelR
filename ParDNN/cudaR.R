# ParallelR:  GTC2016, Unblock Performance limit in R by CUDA
# Author: Peng Zhao

# preload static object file
dyn.load("cudaR.so")

# GPU version of pmax
pmax.cuda <- function(A, threshold, devID=0L)
{
  rst <- .Call("pmax_cuda",
                A,
                threshold,
                as.integer(devID)
	      )
  dim(rst) <- dim(A)
  return(rst)
}

# GPU version of GEMM from cuBLAS
cuBLAS <- function(A, B, transA=FALSE, transB=FALSE, devID=0L) 
{
  tA <- ifelse(transA == FALSE, 0L, 1L)
  tB <- ifelse(transB == FALSE, 0L, 1L)
  rst <- .Call("gemm_cuda",
                A,
		B,
		tA,
		tB,
		as.integer(devID)
	      )
  	  
  row.size <- dim(A)[1]
  col.size <- dim(B)[2]
  if(tA == 1) row.size <- dim(A)[2]
  if(tB == 1) col.size <- dim(B)[1]	  
  dim(rst) <- c(row.size, col.size)
  return(rst)	  
}
