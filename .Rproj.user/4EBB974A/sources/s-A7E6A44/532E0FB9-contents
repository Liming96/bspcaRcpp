//[[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
using namespace std;

// declare ExchangeRcpp_cov
List Exchange_Rcpp(const mat X, vec w, const int sparsity, vec sacrifice, 
                   vec residual, uvec active, const int lambda){
  const int p = X.n_cols;
  vec w_new(p);
  mat obj = trans(w) * X * w - lambda * (trans(w) * w - 1);
  double obj1 = obj(0,0);
  mat rest;
  vec m = sacrifice;
  std::sort(m.begin(), m.end());
  uvec one(p,fill::ones);
  uvec inactive = one - active;
  int k = p - sparsity;
  int c = 1;
  while(c < std::min(sparsity,  p - sparsity)){
    uvec ex1 = sacrifice < m[k+c];
    uvec ex2 = sacrifice >= m[k-c];
    uvec Exch_active = active % ex1;
    uvec Exch_inactive = inactive % ex2;
    uvec uExch_active = one - Exch_active;
    uvec active_new = active % uExch_active + Exch_inactive;
    mat Xp = X.cols(find(active_new > 0));
    mat U,V;
    vec s;
    svd(U,s,V,Xp);
    for(int j = 0,i=0; j < p; j++){
      if(active_new[j]==1){
        w_new[j] = V(i,0);
        i = i + 1;
      }else{
        w_new[j] = 0;
      }
    }
    mat obj_new = trans(w_new) * X * w_new -
      lambda * (trans(w_new) * w_new - 1);
    double obj2 = obj_new(0,0);
    if(obj1 < obj2){
      obj1 = obj2;
      active = active_new;
      w = w_new;
    }
    c = c + 1;
  }
  for(int j = 0; j < p; j++){
    if(active[j]==1){
      residual[j] = 0;
    }else{
      rest = (lambda * w[j] - trans(X.col(j)) * w)/(X(j,j) - lambda);
      residual[j] = rest(0,0);
    }
  }
  for(int j = 0; j < p; j++){
    sacrifice[j] = (lambda - X(j,j)) * pow(w[j] + residual[j],2);
  }
  return List::create(Named("w") = w,
                      Named("sacrifice") = sacrifice,
                      Named("active") = active,
                      Named("obj") = obj1);
};


// [[Rcpp::export]]
NumericVector spca_Rcpp(const NumericMatrix X1, const int sparsity, 
                        NumericVector w1, const int lambda = 10000,
                        const double bess_tol = 1e-3, 
                        const int bess_maxiter = 100){
  mat X = as<mat>(X1);
  int p = X.n_cols;
  vec residual(p), sacrifice(p);
  uvec active(p);
  vec w = as<vec>(w1);
  mat rest;
  int k = p - sparsity;
  for(int j = 0; j < p; j++){
    rest = (lambda * w[j] - trans(X.col(j)) * w)/(X(j,j) - lambda);
    residual[j] = rest(0,0);
    sacrifice[j] = (lambda - X(j,j)) *pow(w[j] + residual[j],2);
  }
  vec m = sacrifice;
  std::sort(m.begin(), m.end());
  active = sacrifice >= m[k];
  mat Xp = X.cols(find(active>0));
  mat U,V;
  vec s;
  svd(U,s,V,Xp);
  for(int j = 0, i=0; j < p; j++){
    if(active[j]==1){
      w[j] = V(i,0);
      i = i + 1;
    }else{
      w[j] = 0;
    }
  }
  double obj_old = 0; 
  double obj;
  int ii = 0;
  while(ii <= bess_maxiter){
    List res = Exchange_Rcpp(X,w,sparsity,sacrifice,residual,active,lambda);
    vec w_new = res["w"];
    vec sacrifice_new = res["sacrifice"];
    uvec active_new = res["active"];
    mat obj1 = trans(w_new) * X * w_new - lambda * 
      (trans(w_new) * w_new - 1);
    obj = obj1(0,0);
    if(abs((obj-obj_old)/obj) < bess_tol){
      break;
    }else{
      ii = ii + 1;
    }
    w = w_new;
    sacrifice = sacrifice_new;
    active = active_new;
    obj_old = obj;
    
  }
  w = normalise(w);
  return wrap(w);
}
