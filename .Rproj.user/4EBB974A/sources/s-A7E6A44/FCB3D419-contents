#' Get Sparse PCs loadings
#'
#' @param x Matrix.
#' @param k Vector or number.
#' @param type "Gram" or "predictor".
#' @param lambda Lagrange multiplier.
#' @param ncomp Number of PCs needed.
#' @param center. If TRUE, centering x.
#' @param scale. If TRUE, scaling x.
#' @param bess_tol Stop condition.
#' @param bess_maxiter Maximal iterate number.
#'
#' @return A dataframe.
#' @examples
#' \dontrun{
#' data("pitprops")
#' result <- bsPCA(pitprops,k=7,type="Gram")
#' }
#' @export
Abspca <- function(x,k=ncol(x),type=c("Gram","predictor"),lambda=10000,
                       ncomp=min(dim(x)),center.=TRUE,scale.=FALSE,bess_tol=1e-3,
                       bess_maxiter=100){
  if(length(k) == 1){
    k <- rep(k,ncomp)
  }else{
    ncomp <- length(k)
  } 
  
  n <- nrow(x)
  p <- ncol(x)
  X <- switch(type,
              predictor = {
                x_temp <- scale(x,center=center.,scale=scale.)
                X <- t(x_temp)%*%x_temp/(n-1)
              },
              Gram = {
                X <- rootmatrix(x)
              }
  )
  
  Xp <- X
  
  svdobj <- svd(X)
  v <- svdobj$v
  totalvariance <- sum((svdobj$d)^2) 
  alpha <- as.matrix(v[,1:ncomp,drop=FALSE])
  
  W <- matrix(0,p, ncomp) 
  sdev <- rep(0,ncomp)
  
  ccs <- seq(ncomp)
  for(cc in ccs){
    w <- spca_Rcpp(Xp,sparsity=k[cc],alpha[,cc],lambda,bess_tol,bess_maxiter)
    W[,cc] <- w 
    #Delation
    Xp <- Xp-X%*%w%*%t(w)
    
    
    if(cc < ncomp && all(abs(Xp)<1e-14)){
      W <- W[,1:cc,drop=FALSE]
      break
    }
  }
  Z <- Xp%*%W
  qrZ <- qr(Z)
  RqrZ <- qr.R(qrZ)
  sdev <- diag(RqrZ)^2
  bspc <- list(sdev=sdev,rotation=W,X = Xp)
  return(bspc)
}

rootmatrix<-function(x){
  x.eigen<-eigen(x)
  d<-x.eigen$values
  d<-(d+abs(d))/2
  v<-x.eigen$vectors
  return (v%*%diag(sqrt(d))%*%t(v))
} 


