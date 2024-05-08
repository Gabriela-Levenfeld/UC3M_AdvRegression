################################################################################
## Project: Advanced Regression and Prediction
##
################################################################################

# Ridge model (alpha=0)
train_and_evaluate_ridge <- function(x_train, y_train, x_test, y_test) {
  ridge_init <- cv.glmnet(x_train, y_train, alpha=alpha_ridge)
  
  lambdaGrid <- 10^seq(log10(ridge_init$lambda[1]), log10(0.1), length.out=150) # log-spaced grid
  
  kcvRidge <- cv.glmnet(x=x_train, y=y_train, alpha=alpha_ridge, nfolds=10, lambda=lambdaGrid)
  best_lambda_ridge <- kcvRidge$lambda.1se
  ridge_res <- summary_mod_performance(x_test, y_test, kcvRidge, alpha_ridge)
  
  output <- list(res=ridge_res, best_lambda=best_lambda_ridge, best_alpha=alpha_ridge)
  return(output)
}

# Lasso model (alpha=1)
train_and_evaluate_lasso <- function(x_train, y_train, x_test, y_test) {
  lasso_init <- cv.glmnet(x_train, y_train, alpha=alpha_lasso)
  
  lambdaGrid <- 10^seq(log10(lasso_init$lambda[1]), log10(0.1), length.out=150) # log-spaced grid
  
  kcvLasso <- cv.glmnet(x=x_train, y=y_train, alpha=alpha_lasso, nfolds=10, lambda=lambdaGrid)
  # Predictions using the lambda.1se
  best_lambda_lasso <- kcvLasso$lambda.1se
  
  lasso_res <- summary_mod_performance(x_test, y_test, kcvLasso, alpha_lasso)
  output <- list(res=lasso_res, best_lambda=best_lambda_lasso, best_alpha=alpha_lasso)
  return(output)
}

# Elastic Net
train_and_evaluate_elasticNet <- function(x_train, y_train, x_test, y_test) {
  # glmnet doesn't allow to tune alpha, that's why we use caret first
  # 1º Search alpha value with caret.
  # 2º Fix alpha value and search lambda with glmnet.
  # 3º Make predictions with both best values.
  elasticNet_init <- caret::train(
    x_train, y_train, method="glmnet",
    preProc=c("zv", "center", "scale"),
    trControl=trainControl(method="cv", number=10), # 10-fold cv
    tuneLength=10 # for the grid
  )
  best_alpha_enet <- elasticNet_init$bestTune$alpha
  
  lambdaGrid <- 10^seq(log10(elasticNet_init$bestTune$lambda), log10(0.1), length.out=150) # log-spaced grid
  kcvEnet <- cv.glmnet(x=x_train, y=y_train, alpha=best_alpha_enet, nfolds=10, lambda=lambdaGrid)
  best_lambda_enet <- kcvEnet$lambda.1se
  
  elasticNet_res <- summary_mod_performance(x_test, y_test, kcvEnet, best_alpha_enet)
  
  output <- list(res=elasticNet_res, best_lambda=best_lambda_enet, best_alpha=best_alpha_enet)
  return(output)
}
