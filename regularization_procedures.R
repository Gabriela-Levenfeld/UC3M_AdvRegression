################################################################################
## Project: Advanced Regression and Prediction
##
## Script for Option 1: Compare the performance of different regularization 
## procedures through a simulation study, constructing an error measure.
##
################################################################################

# Load libraries and functions -------------------------------------------------
library(dplyr)
library(ggplot2)
library(pander)

source("functions/split_data.R")
source("functions/apply_penalty_model.R")

# ------------------------------------------------------------------------------
set.seed(1234) # for reproducibility

# Global variables
n_scenarios <- 6
n_iter <- 120 # could be 100

# Iterate through all scenarios
for (i in 1:n_scenarios) {
  cat("Computing ", i, "scenario.") # For user feedback
  
  # Load data for current scenario
  simulations <- read.csv(paste("syntheticData/dataset", i, ".csv", sep=""))
  aux <- split_data(simulations)
  
  # The following code line must be deactivated just for scenario 6!
  # columns <- c("iteration", "best_lambda", "best_alpha", "MEP", "MSEP", "MAEP", "R2", paste0("X", 0:40))
  columns <- c("iteration", "best_lambda", "best_alpha", "MEP", "MSEP", "MAEP", "R2", "X0", paste0("X", 2:11))
  
  ridge_df <- setNames(data.frame(matrix(ncol=length(columns), nrow=0)), columns)
  lasso_df <- ridge_df
  elasticNet_df <- ridge_df

  # Iterate the simulation
  for (iter in 1:n_iter) {
    cat(iter, " ") # For user feedback
    # Apply regularization techniques
    #   - Ridge model (alpha=0)
    #   - Lasso model (alpha=1)
    ridge_res <- train_and_evaluate_penalized_model(aux$x_train, aux$y_train, aux$x_test, aux$y_test, fixed_alpha=0)
    lasso_res <- train_and_evaluate_penalized_model(aux$x_train, aux$y_train, aux$x_test, aux$y_test, fixed_alpha=1)
    elasticNet_res <- train_and_evaluate_elasticNet(aux$x_train, aux$y_train, aux$x_test, aux$y_test)
    
    # Get the coefficients
    ridge_coefs <- as.numeric(ridge_res$res$coef_values)
    lasso_coefs <- as.numeric(lasso_res$res$coef_values)
    enet_coefs <- as.numeric(elasticNet_res$res$coef_values)
    
    # Append results including coefficients to the corresponding df
    ridge_df[nrow(ridge_df)+1, ] <- c(iter, ridge_res$best_lambda, ridge_res$best_alpha, unlist(ridge_res$res$errors), ridge_coefs)
    lasso_df[nrow(lasso_df)+1, ] <- c(iter, lasso_res$best_lambda, lasso_res$best_alpha, unlist(lasso_res$res$errors), lasso_coefs)
    elasticNet_df[nrow(elasticNet_df)+1, ] <- c(iter, elasticNet_res$best_lambda, elasticNet_res$best_alpha, unlist(elasticNet_res$res$errors), enet_coefs)
  }
  
  # Combine in 1 df
  combined_data <- rbind(
    ridge_df %>% mutate(Method = "Ridge"),
    lasso_df %>% mutate(Method = "Lasso"),
    elasticNet_df %>% mutate(Method = "ElasticNet")
  )
  
  # Save df as csv for later analysis
  write.csv(combined_data, file=paste("results/scenario", i, ".csv", sep=""), row.names=FALSE)
}


# Analysed results -------------------------------------------------------------

## Study MSE measure ---------------------------------------------------------
all_summary_stats <- list()

for (i in 1:n_scenarios) {
  cat("Analysis of scenario ", i, "\n")
  
  # Load Scenario
  data_scenario <- read.csv(paste0("results/scenario", i, ".csv"), sep=",")
  
  # Create boxplot for MSEP
  p <- ggplot(data_scenario, aes(x=Method, y=MSEP, fill=Method)) +
    geom_boxplot() +
    labs(title=paste("Boxplot of MSEP - Scenario", i),
         x="Regularization Method", y="Mean Squared Error of Prediction") +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal()
  
  p
  ggsave(paste0("results/boxplotsMSE/MSEP_boxplot_scenario", i, ".png"), plot=p) # Save the plot
  
  # Print statistics for Latex
  summary_stats <- data_scenario %>%
    group_by(Method) %>%
    summarise(
      Average_Error=mean(MSEP),
      Max_Error=max(MSEP),
      Min_Error=min(MSEP),
      Std_Deviation=sd(MSEP)
    )
  
  # Store them
  all_summary_stats[[i]] <- summary_stats
  pander(summary_stats)
}

save(all_summary_stats, file="all_summary_stats.RData")


# Con los cvs sacar las métricas para el report:
# Error medio y media de las betas



## Benchmark -------------------------------------------------------------------
ols1 <- lm(y~., data = aux$train) # OLS
olsBIC_mod1 <- stepAIC(ols1, k=log(nrow(aux$x_train)), trace=0) # OLS combined with the BIC
## -----------------------------------------------------------------------------
