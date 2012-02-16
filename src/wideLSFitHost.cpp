
#include<R.h>
#include<Rcpp.h>

#include "wideLSFitCoproc.h"
using namespace Rcpp;

//#include<iostream>
using namespace std;

// Computes LS-fit statistics for one- and two-predictor models.
//
RcppExport SEXP WideLSRcpp(SEXP sx, SEXP sy, SEXP syIdx, SEXP sp1, SEXP sp2) {
  NumericMatrix x = as<NumericMatrix>(sx);
  NumericMatrix y = as<NumericMatrix>(sy);
  IntegerVector yIdx = as<IntegerVector>(syIdx);
  List preds = as<List>(sp1);
  IntegerVector keep = as<IntegerVector>(sp2);

  // R is built in shared memory, once per regression, and discarded.
  // There is one column for the intercept, plus one for each term.
  //
  int numPreds = preds.length();
  int rDim = accumulate(keep.begin(), keep.end(), 0) + 1;

  // T-score vectors returned.
  //
  IntegerVector p1 = as<IntegerVector>(preds[0]);
  IntegerVector p2;
  NumericMatrix tScores(rDim, p1.length());
  NumericMatrix coefs(rDim, p1.length());

  // For now, uses ad-hoc method to select among the three model
  // shapes supported.  A more general scheme will pack the terms
  // into a table.
  //
  if (numPreds > 1)
    p2 = as<IntegerVector>(preds[1]);

  switch(rDim) {
  case 2:
    WideLSF2(x.begin(), x.ncol(), x.nrow(), y.begin(), y.ncol(), yIdx.begin(), p1.length(), coefs.begin(), tScores.begin(), p1.begin());
    break;
  case 3:
    WideLSF2(x.begin(), x.ncol(), x.nrow(), y.begin(), y.ncol(), yIdx.begin(), p1.length(), coefs.begin(), tScores.begin(), p1.begin(), p2.begin());
    break;
  case 4:
    WideLSF2(x.begin(), x.ncol(), x.nrow(), y.begin(), y.ncol(), yIdx.begin(), p1.length(), coefs.begin(), tScores.begin(), p1.begin(), p2.begin(), 1);
    break;
  default: // Will exit without filling in 'tScores'.
    //    cout << "Should not be here"  << endl;
    break;
  }

  return List::create(Named("coef") = coefs, Named("tscore") = tScores);
}


