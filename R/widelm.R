
wideLM <- function (x, y, yIdx, preds, formulaMode=c("additive", "accumulated", "reduced", "saturated"), addendum = NULL) {
  if (nrow(x) != nrow(y))
    stop("Dimensions of response and design do not agree")

  # TODO:  Implementation-dependent parameter to be encapsulated.
  #
  if (nrow(x) > 1024)
    stop("Observation count exceeds hardware capabilities")

  numPreds <- length(preds)
  if (numPreds > 2)
    stop("WideLM currently limited to two predictors")

  fm=match.arg(formulaMode)
  if (!is.null(addendum)) {
    if (fm == "additive" || fm == "saturated")
      message("addendum ignored in ", fm, " mode")    
  }
  else {
    if (fm == "reduced")
      message("null addendum, reverting to saturated mode")
    else if (fm == "accumulated")
      message("null addendum, reverting to additive mode")
  }
  
  # "keepTerms" records the terms to be modelled using a lexical ordering
  # to index term positions.  Intuitively, for predictor list 'preds'
  # containing predictors 'p_1', 'p_2', ..., 'p_numPreds', the ordering
  # is as follows:
  #
  # The first 'numPreds' indices refer to the individual predictors:
  # Indices 1, ..., numPreds refer to 'p_1', 'p_2', ... 'p_numPreds'.
  #
  # The next numPreds-choose-two indices refer to pairs:
  #
  # The next 'num_preds-1' indices to 'p_1 * p_2', ... , 'p_1 * p_numpreds'.
  # The next 'num_preds-2' to 'p_2 * p_3, ..., p_2 * p_numpreds'.
  # .. and so forth until all pairs have been enumerated.
  #
  # The next numPreds-choose-three indices refer to triples:
  # p_1 * p_2 * p_3, p_1 * p_2 * p_4, ..., p_1 * p_2 * p_numpreds
  # p_2 * p_3 * p_4, ..., p_2 * p_3 * p_numpreds
  # and so forth until all triples have been enumerated.
  #
  # This scheme is repeated until the single n-fold product is enumerated.
  # This accounts for 2^numpreds - 1 indices.  The final index is reserved
  # to hold a zero.
  #
  # It is assumed that all 'numPreds' predictors are included in the model,
  # so the values assigned to the first 'numPreds' positions are always
  # unity.
  #
  totTerms <- 2^numPreds
  keepTerms = switch(fm,
    "additive" = c(rep(1,numPreds),rep(0, totTerms-numPreds)),
    "accumulated" = setAddendum(c(rep(1, numPreds),rep(0, totTerms-numPreds)), numPreds, addendum, 1),
    "reduced" = setAddendum(rep(1, totTerms), numPreds, addendum, 0),
    "saturated" = rep(1, totTerms)
    )
  keepTerms[totTerms] <- 0

  # Translates index vectors to zero-based for C++:
  #
  preds <- lapply(preds, function(v) {v - 1})
  yIdx <- yIdx - 1

  output <- .Call("WideLSRcpp", x, y, yIdx, preds, keepTerms)
  output[c("coef", "tscore")]
}

# Sets or unsets the predictors specified in the integer vector 'add'.
# Returns the final predictor set.
#
setAddendum <- function(ktIn, numPreds, add, val) {
  kt <- ktIn
  add <- add + numPreds
  for (i in 1:length(add)) {
    if (add[i] <= numPreds || add[i] >= length(kt))
      stop("invalid addendum index ")
    kt[add[i]] <- val
  }
  kt
}
