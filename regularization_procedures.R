################################################################################
## Project: Advanced Regression and Prediction
##
## Script for Option 1: Compare the performance of different regularization 
## procedures through a simulation study, constructing an error measure.
##
################################################################################

# Load libraries and functions -------------------------------------------------
library(MASS)
library(glmnet)
library(caret)
library(dplyr)

source("functions/split_data.R")
source("functions/summary_mod_performance.R")
source("functions/apply_penalty_model.R")

set.seed(1234) # for reproducibility

# Global variables
alpha_ridge <- 0
alpha_lasso <- 1

n_scenarios <- 5
n_iter <- 10

# Iterate through all scenarios
for (i in 1:n_scenarios) {
  cat("Computing ", i, "scenario.") # For user feedback
  
  # Load data for current scenario
  simulations <- read.csv(paste("syntheticData/scenario", i, ".csv", sep=""))
  aux <- split_data(simulations)
  
  # Names for the columns for the csv file
  columns <- c("iteration", "best_lambda", "best_alpha", "MEP", "MSEP", "MAEP", "R2", paste0("X", 0:10))
  ridge_df <- setNames(data.frame(matrix(ncol=length(columns), nrow=0)), columns)
  lasso_df <- ridge_df
  elasticNet_df <- ridge_df

  # Iterate the simulation
  for (iter in 1:n_iter) {
    # Apply regularization techniques
    ridge_res <- train_and_evaluate_ridge(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    lasso_res <- train_and_evaluate_lasso(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    elasticNet_res <- train_and_evaluate_elasticNet(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    
    # Get the coefficients
    ridge_coefs <- as.numeric(ridge_res$res$coef_values@x)
    lasso_coefs <- as.numeric(lasso_res$res$coef_values@x)
    enet_coefs <- as.numeric(elasticNet_res$res$coef_values@x)
    
    # Append results including coefficients to the corresponding dataframe
    ridge_df[nrow(ridge_df)+1, ] <- c(iter, ridge_res$best_lambda, ridge_res$best_alpha, unlist(ridge_res$res$errors), ridge_coefs)
    lasso_df[nrow(lasso_df)+1, ] <- c(iter, lasso_res$best_lambda, lasso_res$best_alpha, unlist(lasso_res$res$errors), lasso_coefs)
    elasticNet_df[nrow(elasticNet_df)+1, ] <- c(iter, elasticNet_res$best_lambda, elasticNet_res$best_alpha, unlist(elasticNet_res$res$errors), enet_coefs)
  }
  
  # Save df for analysed results later
  write.csv(ridge_df, file=paste("results/scenario", i, "_Ridge.csv", sep=""), row.names=FALSE)
  write.csv(lasso_df, file=paste("results/scenario", i, "_Lasso.csv", sep=""), row.names=FALSE)
  write.csv(elasticNet_df, file=paste("results/scenario", i, "_ElasticNet.csv", sep=""), row.names=FALSE)
}





# Con los cvs sacar las métricas para el report:
# Error medio y media de las betas
# Plots (?)


## Benchmark -------------------------------------------------------------------
ols1 <- lm(y~., data = aux1$train) # OLS
olsBIC_mod1 <- stepAIC(ols1, k=log(nrow(aux1$x_train)), trace=0) # OLS combined with the BIC
## -----------------------------------------------------------------------------

# To remove --------------------------------------------------------------------
# Function that works but does not store anything.
# Too heavy in memory 
for (i in 1:n_scenarios) {
  # Load data for current scenario
  simulations <- read.csv(paste("syntheticData/scenario", i, ".csv", sep=""))
  aux <- split_data(simulations)
  
  # Iterate the simulation
  for (iter in 1:n_iter) {
    ridge_results[[iter]] <- train_and_evaluate_ridge(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    lasso_results[[iter]] <- train_and_evaluate_lasso(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    elasticNet_results[[iter]] <- train_and_evaluate_elasticNet(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
  }
  
  # Store scenario results
  scenario_results[[i]] <- list(ridge=ridge_results, lasso=lasso_results, elasticNet=elasticNet_results)
}


# EXTRA  STUFF -----------------------------------------------------------------
# Store results on a csv.
# 3 files for each scenario: One for Ridge, other for Lasso and another for ElasticNet
for (i in 1:n_iter){
  # Just to see how to acces to the values
  
  # Best values of each models selected by tunning
  print(scenario_results[[1]]$ridge[[i]]$best_lambda)
  print(scenario_results[[1]]$ridge[[i]]$best_alpha)
  # Error
  print(scenario_results[[1]]$ridge[[i]]$res$errors["MEP"])
  print(scenario_results[[1]]$ridge[[i]]$res$errors["MSEP"])
  print(scenario_results[[1]]$ridge[[i]]$res$errors["MAEP"])
  print(scenario_results[[1]]$ridge[[i]]$res$errors["R2"])
  # Beta coeff
  print(scenario_results[[1]]$ridge[[i]]$res$coef_values@Dimnames[[1]])
  print(scenario_results[[1]]$ridge[[i]]$res$coef_values@x)
}
