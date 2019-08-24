// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// spca_Rcpp
NumericVector spca_Rcpp(const NumericMatrix X1, const int sparsity, NumericVector w1, const int lambda, const double bess_tol, const int bess_maxiter);
RcppExport SEXP _bspcaRcpp_spca_Rcpp(SEXP X1SEXP, SEXP sparsitySEXP, SEXP w1SEXP, SEXP lambdaSEXP, SEXP bess_tolSEXP, SEXP bess_maxiterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const NumericMatrix >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< const int >::type sparsity(sparsitySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w1(w1SEXP);
    Rcpp::traits::input_parameter< const int >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const double >::type bess_tol(bess_tolSEXP);
    Rcpp::traits::input_parameter< const int >::type bess_maxiter(bess_maxiterSEXP);
    rcpp_result_gen = Rcpp::wrap(spca_Rcpp(X1, sparsity, w1, lambda, bess_tol, bess_maxiter));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bspcaRcpp_spca_Rcpp", (DL_FUNC) &_bspcaRcpp_spca_Rcpp, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_bspcaRcpp(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
