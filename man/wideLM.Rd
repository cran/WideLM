% File man/wideLM.Rd
% Part of the widelm package

\name{wideLM}
\alias{wideLM}
\concept{regression}
\title{Fitting Multiple Models of Modest Size}
\description{
  Fits numerous linear models to a fixed design and response.  The
  predictors are specified via a list of parallel index vectors and
  the interaction terms are computed on the fly.  The design matrix is
  currently restricted to contain 1024 rows or less.  A vector of
  indices into the columns of the response matrix is used to specify
  the response to be fit.
}

\usage{
wideLM(x, y, yIdx, preds, formulaMode, addendum=NULL)
}

\arguments{
  \item{x}{the design matrix.}
  \item{y}{a matrix of responses. The responses are organized as
  columns.  Hence \code{x} and \code{y} should have the same number of rows.}
  \item{yIdx}{a vector of column indices into the response matrix}
  \item{preds}{A list of vectors of column indices into the design
  matrix.  Together, these vectors specify the predictors to be fit in
  each individual model.  All vectors should have the same length, that
  is, the total number of models to be tested.
  N.B.:  The package is currently limited to two predictors.}
\item{formulaMode}{"additive", "saturated", "accumulated", "reduced".
  The terms \code{additive} and \code{saturated} refer to types of
  models.  The terms \code{accumulated} and \code{reduced} are shorthand
  for additions to the \code{additive} and deletions from the
  \code{saturated} model, respectively, using the \code{addendum}
  mechanism described below.}
  \item{addendum}{a vector of exceptions to the additive model or
    saturated model. Exceptions give a compact way to describe models
    somewhat larger than the additive model or somewhat smaller than the
    saturated model.  Exceptions are integers which enumerate the possible
    interaction terms in a saturated model on 'n' predictors.  Hence
    their values range from 1 to 2^n - (n+1).

    Exceptions only apply to the \code{accumulated} and \code{reduced}
    modes. For the \code{accumulated} mode, all enumerated interaction
    terms are appended to the additive model.  For the \code{reduced}
    mode, the enumerated interaction terms are deleted from the
    saturated model.
    
    The actual mapping is based on the lexical ordering of subsets of
    a set of size 'n'.  This is explained intuitively via following
    schema, for predictors 'p_1', 'p_2', ..., 'p_n':
    
    The numbers from 1 to (n-choose-two) name interaction pairs:
  
      1, ..., n-1 refer to 'p_1 * p_2', ... ,'p_1 * p_n', respectively.

      The next 'n-2' numbers refer to 'p_2 * p_3' , ..., 'p_2 * p_n'
      
    ... and so forth until all pairs have been enumerated.
  
    The next (n-choose-three) numbers refer to triples:
    
     'p_1 * p_2 * p_3', 'p_1 * p_2 * p_4', ..., 'p_1 * p_2 * p_n'
 
     'p_2 * p_3 * p_4', ..., 'p_2 * p_3 * p_n'

    ... and so forth until all triples have been enumerated.
  
    This scheme is repeated until the single n-fold product is
    enumerated, which should have value 2^n - (n+1).
  }
}

\value{A list containing the matrices \code{coef} of coefficients
  and \code{tscore} of t-scores.  The columns refer to the indices
  of the vectors given in the \code{preds} list.  The rows are
  ordered as follows:

    intercept, predictors, interaction terms (in lexical order).
}

\author{
  Mark Seligman at Insilicos LLC.  Original Fortran implementation
  computing the t-scores was developed by Chris Fraley at Insilicos LLC.
  }
