################################################################################
## Project: Advanced Regression and Prediction
##
## Function that summarizes the performance of a model by providing:
##  - Coefficients values to evaluate shrinkage methods, and
##  - Compute accuracy metrics based on prediction error for unseen data.
##
##
## Note: This function used the lambda.1se for compute the results.
##
################################################################################

source("functions/error_metrics.R")

summary_mod_performance <- function(x_test, y_test, model, alpha_value) {
  # Predictions and Errors using the lambda.1se
  final_mod <- model$glmnet.fit
  best_lambda_1se <- model$lambda.1se # Take the best lambda from the k-fold CV
  
  pred_coef <- predict(final_mod, type="coefficients", s=best_lambda_1se, alpha=alpha_value)
  pred_y <- predict(final_mod, newx=x_test, s=best_lambda_1se, alpha=alpha_value)
  errors <- round(error_metrics(pred_y, y_test), 3) # Select just the 3 first decimals
  
  # For set as zero the non-selected variable in order to store them
  pred_coef <- as.vector(pred_coef)

  output <- list(coef_values=pred_coef, pred_y=pred_y, errors=errors)
  
  return(output)
}
