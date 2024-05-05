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
  return(c(
    ME = mean(err), # Mean error of prediction
    MSE = mean(err^2), # Mean squared error of prediction
    MAE = mean(abs(err)) # Mean absolute error of prediction
  ))
}
