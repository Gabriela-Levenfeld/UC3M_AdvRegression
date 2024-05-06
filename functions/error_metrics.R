################################################################################
## Project: Advanced Regression and Prediction
##
## Function to compute some error metrics (ME, MSE, and MAE), in order to
## evaluate the performance of the simulated data scenarios.
##
## Provided by: Silvia Novo, professor of this subject.
##
################################################################################

error_metrics <- function(pred, obs, na.rm=FALSE, tol=sqrt(.Machine$double.eps)) {
  err <- obs - pred # Errores
  if(na.rm) {
    is.a <- !is.na(err)
    err <- err[is.a]
    obs <- obs[is.a]
  }
  
  SS_res <- sum(err^2)
  SS_tot <- sum((obs - mean(obs))^2)
  R_squared <- 1 - SS_res/SS_tot
  
  return(c(
    MEP = mean(err), # Mean error of prediction
    MSEP = mean(err^2), # Mean squared error of prediction
    MAEP = mean(abs(err)), # Mean absolute error of prediction
    R2 = R_squared # R-squared
  ))
}
