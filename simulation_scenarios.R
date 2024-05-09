################################################################################
## Project: Advanced Regression and Prediction
##
## This script create the simulation study, we have designed five different 
## scenarios to later study the regularization techniques we have seen during 
## lectures.
##
################################################################################

# Load required libraries and files --------------------------------------------
source("functions/generate_synthetic_data.R")

# Set up all common parameters used --------------------------------------------
set.seed(1234) # for reproducibility
n <- 240 # number of observations
p <- 10 # nums of predictors

# Generate random values for the betas (as uniform integers in [-10,10])
beta_coef <- runif(p+1, min=-10, max=10) # Include beta_0 for the intercept

print('Values of the beta coefficients:')
for (i in 1:(p+1)){
  print(beta_coef[i])
}


# Scenario 1 -------------------------------------------------------------------

## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   No correlation between predictors, X
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)

X_params1 <- list(X_mu=0, X_sigma=1)
noise_params1 <- list(mean=0, sd=2)
data1 <- generate_data_indep(n, p, beta_coef,
                             X_matrix=NULL, X_params1,
                             noise_type="normal", noise_params1)

## Check data ------------------------------------------------------------------
mcor1 <- cor(data1)
mcor1
abs(mcor1)>0.8
corrplot::corrplot(mcor1, method = "number")

# Another option for the correlation matrix
# library(GGally)
# ggpairs(data_without_intercept,
#         aes(fill="pink"),
#         lower = list(continuous = "points", combo = "box_no_facet"),
#         upper = list(continuous = "cor"),
#         diag = list(continuous = "barDiag"),
#         title = "Scatter plot matrix"
# )


# Scenario 2 ------------------------------------------------------------------
# To check robustness of the methods
## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   No correlation between predictors, X
#   Same as in scenario 1
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> Exponential
# Betas coefficients are the same as in Scenario 1

noise_params2 <- list(rate=0.35) # lamdba parameter of the exp. distribution
data2 <- generate_data_indep(n, p, beta_coef, 
                             # For reused scenario 1 features
                             X_matrix=subset(data1, select = -c(y)),
                             X_params=X_params1, 
                             noise_type="exponential", noise_params2)

## Check data ------------------------------------------------------------------
# Correlation between the predictors and the output change (but not too much)
mcor2 <- cor(data2)
mcor2
abs(mcor2)>0.8
corrplot::corrplot(mcor2, method = "number")

# Scenario 3 ------------------------------------------------------------------
## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   No correlation between predictors, X
#   Same as in scenario 1
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> Normal, but changing the standard deviation
# Betas coefficients are the same as in Scenario 1

noise_params3 <- list(mean=0, sd=0.05) # decrease the sd
data3 <- generate_data_indep(n, p, beta_coef,
                             X_matrix=NULL, X_params1,
                             noise_type="normal", noise_params3)

## Check data ------------------------------------------------------------------
# Correlation between the predictors and the output change (but not too much)
mcor3 <- cor(data3)
mcor3
abs(mcor3)>0.8
corrplot::corrplot(mcor3, method = "number")


# Scenario 4 ------------------------------------------------------------------

## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   No correlation between predictors, X
#   Same as in scenario 1
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients: half of them are equal to zero

beta_coef_half_zeros <- beta_coef
zero_index <- c(2, 3, 8, 9, 10)
beta_coef_half_zeros[zero_index] <- 0

data4 <- generate_data_indep(n, p, beta_coef_half_zeros,
                             X_matrix=NULL, X_params1,
                             noise_type="normal", noise_params1)

## Check data ------------------------------------------------------------------
mcor4 <- cor(data4)
mcor4
abs(mcor4)>0.8
corrplot::corrplot(mcor4, method = "number")

# Scenario 5 ------------------------------------------------------------------
#https://medium.com/@marc.jacobs012/drawing-and-plotting-observations-from-a-multivariate-normal-distribution-using-r-4c2b2f64e1a3

## Generate synthetic data -----------------------------------------------------
# Multicolinearity
#   X are positive correlated
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients are the same as in Scenario 1

rho <- 0.8 # Positive correlation coefficient
correlation_groups <- list(c(1, 2, 3), c(6, 7)) # Define correlation groups

set.seed(12345)
data5 <- generate_data_blocks(n, p)
# TODO: Create negative correlation on the data

## Check data ------------------------------------------------------------------
mcor5 <- cor(data5)
mcor5
abs(mcor5)>0.8
corrplot::corrplot(mcor5, method = "number")


# Scenario 6 ------------------------------------------------------------------
# https://watermark.silverchair.com/jrsssb_67_2_301.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2MwggNfBgkqhkiG9w0BBwagggNQMIIDTAIBADCCA0UGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMHvbXDzKU7FNlJ1v3AgEQgIIDFgsu4ebwEuKZpsHFmC3KIVjnHYHlD32OpF8svtc5UrPVwBAcBRkAIsgi579GOK_bnkVRnH5rhT-rWHycJYRXSnCndspVTaZnDJBJUJto1WMyVgOtO7-3rCViTjxlg2ZMd9z67sP3v1nsr6A29Gk2Jsl3Psq9TXkR0osrpqgVTSYwAEaUNBCI8nMRt2F_ZCTZ_mg-LRDQiM6AiBmKg5x3ZjH2meymIjC3pYgWGT7swTeAuhuznZPXVs9n0AhaD7C0mIt-22RPgiR2Wd6l2IQwq-HXC_yUQxyPerr6q0UAM5jWZIZzajashvr3X0gHow_PRldGzo8zEJOuqJnG5SRsS_FEAB_SSsyf1u3HPL9HEjk7iVBFP7kmmIsHDdKTF-zy47iTyXhnrboHJ271mIA5PgvW05X0pkjvIoN81YbMe07bwNEhVcDnQPGXkNT8yZcBhMQIjuqEB_ZtsP9mcwZpjKPU8J8kSg2S0-RZagx6IB2KHb7C01ae_pw2jDV13VUQTRUR6TasE9m67G0Vy5273bPH9ByNrWXRaMMoW9-vA1YnC_LaXhePSgQ78v-qpTockMVUj6wVE3yCNcgaaNLXmmcXPgy-s7szRr1u61csuoHEOSRHKaOCLHhehf168p9k3vIcGSPD36QP4FFMvYzLF1sBa9yDHwDsTIAbjrWNB4XGtzTZbk5fNQvVCltitNZR-ceCCeNM4ZfFrEMtPd2C9sdWLSJeYU5eQszdZMetPYLe5wbAzjcH10DN8v5pG0dvijgeVqD0WdfmMmPSFAm6hOJd7yU8M21ORZaPRUUZpqp-_YfVnRrwFM9v9wfbCHpbZqxbtkB5MdN0OX2EhgyLuQFFhvUBjzTbO1y-ydVpLEnUolJIKNHtrcE_dwpD2_IV4u88xXoLshSHNjUHBwdItflhDbpVWiFY5A1BXjRD7_hmn0qfIhqt3-TO2AKdu3Nucv9nwpKksPqVZZHeapC-3LWK4RTLy6c1QWxyv1KHywrIeGIjnBjVMZAoVbJDGye0nd5osfphS1F9XhfWXJjTsI2JP7aP5Yo

## Generate synthetic data -----------------------------------------------------
# High dimensional data
# p > n
# Each observation is drawn independently
# X (predictors) matrix:
#   No correlation between predictors, X
#   Same as in scenario 1
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients are the same as in Scenario 1

set.seed(1234) # for reproducibility
n_new <- 30
p_new <- 40
beta_coef_new <- runif(p_new+1, min=-10, max=10)

data6 <- generate_data_indep(n_new, p_new, beta_coef_new,
                             X_matrix=NULL, X_params1,
                             noise_type="normal", noise_params1)

## Check data ------------------------------------------------------------------
mcor6 <- cor(data6)
mcor6
abs(mcor6)>0.8
corrplot::corrplot(mcor6, method = "number")


# We can also used the circle representation, more visually.
# But for better understanding is better the number representation.
# corrplot::corrplot(mcor1, method = "circle")


# Save stuff -------------------------------------------------------------------
# 1 scenario - 1 file
write.csv(data1, "syntheticData/scenario1_miki.csv", row.names=FALSE)
write.csv(data2, "syntheticData/scenario2_miki.csv", row.names=FALSE)
write.csv(data3, "syntheticData/scenario3_miki.csv", row.names=FALSE)
write.csv(data4, "syntheticData/scenario4_miki.csv", row.names=FALSE)
write.csv(data5, "syntheticData/scenario5_miki.csv", row.names=FALSE)
write.csv(data6, "syntheticData/scenario6_miki.csv", row.names=FALSE)

# Save both beta coefficients used
beta_data <- data.frame(
  original_beta=beta_coef,
  half_zero_beta=beta_coef_half_zeros
)
write.csv(beta_data, "syntheticData/beta_coefficients_miki.csv", row.names=FALSE)








# Function not working
generate_data_group_correlation <- function(n, p, beta_coef, correlation_groups, noise_params) {
  # 1. Feature Matrix
  Sigma <- diag(p) # Identity matrix to start with
  
  # Generate correlated feature matrix for specified groups
  for (group in correlation_groups) {
    group_size <- length(group)
    group_Sigma <- matrix(1, nrow=group_size, ncol=group_size) # All ones for correlation within group
    diag(group_Sigma) <- 1 # Set the diagonal to 1 (correlation with itself)
    Sigma[group, group] <- group_Sigma
  }
  
  # Generate uncorrelated feature matrix for remaining variables
  uncorrelated_vars <- setdiff(1:p, unlist(correlation_groups))
  Sigma[uncorrelated_vars, uncorrelated_vars] <- 0
  
  # Generate correlated feature matrix
  X_correlated <- mvrnorm(n, mu=rep(0, p), Sigma=Sigma)
  X_0 <- matrix(rep(1, n), ncol=1) # Add intercept
  X <- cbind(X_0, X_correlated)
  
  # 2. Creating Linear Relationship
  y_lin <- X %*% beta_coef
  
  # 3. Introducing Random Error (Noise) -> Follows a Normal distribution
  normal_error <- rnorm(n, mean=noise_params$mean, sd=noise_params$sd)
  y <- y_lin + normal_error
  
  # 4. Final Outputs
  data <- data.frame(y=y, X)
  data <- subset(data, select=-c(X1)) # Get rid off the intercept, no need anymore
  
  return(data)
}




# Example usage:
beta_coef <- c(2, 3, 1, rep(0, p)) # Coefficients for linear relationship (e.g., 2*y1 + 3*y2 + y3)
correlation_groups <- list(c(1, 2, 3), c(6, 7)) # Define correlation groups
noise_params <- list(mean=0, sd=1) # Noise parameters

data <- generate_data_group_correlation(n, p, beta_coef, correlation_groups, noise_params)
mcor <- cor(data)
corrplot::corrplot(mcor, method = "number")


generate_data_blocks <- function(n, p, block_size = 3) {
  X <- matrix(rnorm(n * p), ncol = p)
  
  noise_vector_first = rnorm(n)  # Common noise vector for the first block
  for (j in 1:block_size) {
    X[, j] <- X[, j] + noise_vector_first
  }
  
  # Check if a second block is feasible before proceeding
  if (2 * block_size <= p) {
    noise_vector_second = rnorm(n)  # Common noise vector for the second block
    for (j in 1:block_size) {
      X[, block_size + j] <- X[, block_size + j] + noise_vector_second
    }
  }
  
  beta <- rnorm(p)
  y <- X %*% beta + rnorm(n)
  
  data <- data.frame(y=y, X)
  
  #return(list(X=X,y=y))
  return(data)
}

data <- generate_data_blocks(n, p)
mcor <- cor(data)
corrplot::corrplot(mcor, method = "number")



# To do list:
# Corregir generation of correlated data
# Dataset 3 y 4 hay que volver a generarlos porque usan la funcion de correlated data
# Eliminar escenario 5, no tiene sentido (no es lo que buscamos)
# Actualizar funciones y ficheros .csv
