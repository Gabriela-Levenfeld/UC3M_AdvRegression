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

## Generate synthetic data -----------------------------------------------------
# Multicolinearity
#   X are correlated
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients: follows a normal distribution

set.seed(12345)
data5 <- generate_correlated_data_groups(n, p)
beta_coef_normal <- data5$beta_coef
data5 <- subset(data5, select=-c(beta_coef))

## Check data ------------------------------------------------------------------
mcor5 <- cor(data5)
mcor5
abs(mcor5)>0.8
corrplot::corrplot(mcor5, method = "number")


# Scenario 6 ------------------------------------------------------------------

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
write.csv(data1, "syntheticData/dataset1.csv", row.names=FALSE)
write.csv(data2, "syntheticData/dataset2.csv", row.names=FALSE)
write.csv(data3, "syntheticData/dataset3.csv", row.names=FALSE)
write.csv(data4, "syntheticData/dataset4.csv", row.names=FALSE)
write.csv(data5, "syntheticData/dataset5.csv", row.names=FALSE)
write.csv(data6, "syntheticData/dataset6.csv", row.names=FALSE)
