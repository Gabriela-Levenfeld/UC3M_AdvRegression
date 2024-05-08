################################################################################
## Project: Advanced Regression and Prediction
##
## This script trains and evaluates three types of penalized regression models
## on a given dataset: Ridge Regression, Lasso Regression, and Elastic Net.
##
################################################################################

library(glmnet)
library(caret)
source("functions/summary_mod_performance.R")

train_and_evaluate_penalized_model <- function(x_train, y_train, x_test, y_test, fixed_alpha) {
  init_model <- cv.glmnet(x_train, y_train, alpha=fixed_alpha)
  
  # To solve potential issue where best lambda is on at the grid's extreme
  lambdaGrid <- 10^seq(log10(init_model$lambda[1]), log10(0.1), length.out=150) # log-spaced grid
  
  kcvModel <- cv.glmnet(x=x_train, y=y_train, alpha=fixed_alpha, nfolds=10, lambda=lambdaGrid) # 10-fold cv
  best_lambda <- kcvModel$lambda.1se # Predictions using the lambda.1se
  model_res <- summary_mod_performance(x_test, y_test, kcvModel, fixed_alpha)
  
  output <- list(res=model_res, best_lambda=best_lambda, best_alpha=fixed_alpha)
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
  best_lambda_enet <- kcvEnet$lambda.1se # Predictions using the lambda.1se
  elasticNet_res <- summary_mod_performance(x_test, y_test, kcvEnet, best_alpha_enet)
  
  output <- list(res=elasticNet_res, best_lambda=best_lambda_enet, best_alpha=best_alpha_enet)
  return(output)
}
