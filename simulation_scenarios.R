################################################################################
## Project: Advanced Regression and Prediction
##
## This script create the simulation study, we have designed five different 
## scenarios to later study the regularization techniques we have seen during 
## lectures.
##
################################################################################

# Load required libraries and files --------------------------------------------
library(factoextra) # For PCA and its visualization
library(ggplot2) # For saving plots
source("functions/generate_synthetic_data.R")

# Set up all common parameters used --------------------------------------------
set.seed(1234) # for reproducibility
n <- 240 # nums of observations, must be divisible by 3! (for scenario 5)
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
#https://medium.com/@marc.jacobs012/drawing-and-plotting-observations-from-a-multivariate-normal-distribution-using-r-4c2b2f64e1a3

## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   X are positive correlated
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients are the same as in Scenario 1

rho <- 0.5 # Positive correlation coefficient
data3 <- generate_data_correlated(n, p, beta_coef, rho=rho, noise_params1)
# TODO: Create negative correlation on the data

## Check data ------------------------------------------------------------------
mcor3 <- cor(data3)
mcor3
abs(mcor3)>0.8
corrplot::corrplot(mcor3, method = "number")


# Scenario 4 ------------------------------------------------------------------

## Generate synthetic data -----------------------------------------------------
# Each observation is drawn independently
# X (predictors) matrix:
#   X are positive correlated
#   Follows a Normal distribution (0, 1)
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients: half of them are equal to zero

beta_coef_half_zeros <- beta_coef
zero_index <- c(2, 3, 8, 9, 10)
beta_coef_half_zeros[zero_index] <- 0

data4 <- generate_data_correlated(n, p, beta_coef_half_zeros, rho=rho, noise_params1)

## Check data ------------------------------------------------------------------
mcor4 <- cor(data4)
mcor4
abs(mcor4)>0.8
corrplot::corrplot(mcor4, method = "number")


# Scenario 5 ------------------------------------------------------------------
# https://watermark.silverchair.com/jrsssb_67_2_301.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2MwggNfBgkqhkiG9w0BBwagggNQMIIDTAIBADCCA0UGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMHvbXDzKU7FNlJ1v3AgEQgIIDFgsu4ebwEuKZpsHFmC3KIVjnHYHlD32OpF8svtc5UrPVwBAcBRkAIsgi579GOK_bnkVRnH5rhT-rWHycJYRXSnCndspVTaZnDJBJUJto1WMyVgOtO7-3rCViTjxlg2ZMd9z67sP3v1nsr6A29Gk2Jsl3Psq9TXkR0osrpqgVTSYwAEaUNBCI8nMRt2F_ZCTZ_mg-LRDQiM6AiBmKg5x3ZjH2meymIjC3pYgWGT7swTeAuhuznZPXVs9n0AhaD7C0mIt-22RPgiR2Wd6l2IQwq-HXC_yUQxyPerr6q0UAM5jWZIZzajashvr3X0gHow_PRldGzo8zEJOuqJnG5SRsS_FEAB_SSsyf1u3HPL9HEjk7iVBFP7kmmIsHDdKTF-zy47iTyXhnrboHJ271mIA5PgvW05X0pkjvIoN81YbMe07bwNEhVcDnQPGXkNT8yZcBhMQIjuqEB_ZtsP9mcwZpjKPU8J8kSg2S0-RZagx6IB2KHb7C01ae_pw2jDV13VUQTRUR6TasE9m67G0Vy5273bPH9ByNrWXRaMMoW9-vA1YnC_LaXhePSgQ78v-qpTockMVUj6wVE3yCNcgaaNLXmmcXPgy-s7szRr1u61csuoHEOSRHKaOCLHhehf168p9k3vIcGSPD36QP4FFMvYzLF1sBa9yDHwDsTIAbjrWNB4XGtzTZbk5fNQvVCltitNZR-ceCCeNM4ZfFrEMtPd2C9sdWLSJeYU5eQszdZMetPYLe5wbAzjcH10DN8v5pG0dvijgeVqD0WdfmMmPSFAm6hOJd7yU8M21ORZaPRUUZpqp-_YfVnRrwFM9v9wfbCHpbZqxbtkB5MdN0OX2EhgyLuQFFhvUBjzTbO1y-ydVpLEnUolJIKNHtrcE_dwpD2_IV4u88xXoLshSHNjUHBwdItflhDbpVWiFY5A1BXjRD7_hmn0qfIhqt3-TO2AKdu3Nucv9nwpKksPqVZZHeapC-3LWK4RTLy6c1QWxyv1KHywrIeGIjnBjVMZAoVbJDGye0nd5osfphS1F9XhfWXJjTsI2JP7aP5Yo

## Generate synthetic data -----------------------------------------------------
# Observation are grouped
# X (predictors) matrix:
#   X are grouped
#   Different means for each group in order to group them
# Random Error (Noise) -> N(mean=0, sd=2)
# Betas coefficients are the same as in Scenario 1

num_groups <- 3 # For creating 3 groups -> 80 obvs per group
# Means for each group in order to simulate groups
group_means <- matrix(c(rep(-1.5,p), rep(1,p), rep(3,p)), ncol=p, byrow=TRUE)

# Combine group data into one data frame
data5 <- generate_data_byGroups(n, p, beta_coef, 
                                num_groups, group_means,
                                noise_params1)

pca_data <- data5
data5 <- subset(data5, select=-c(group))

## Check data ------------------------------------------------------------------
mcor5 <- cor(data5)
mcor5
abs(mcor5)>0.8
corrplot::corrplot(mcor5, method = "number")

# Display the group creation
# PCA: excluding group (variable 1) and y (variable 2)
pca_2_vars <- prcomp(pca_data[, -c(1,2)], scale.=TRUE)
# Biplot PCA
pca_plot <- fviz_pca_ind(pca_2_vars,
                         col.ind = pca_data$group, # Color by group
                         palette = c("#4682B4", "#32CD32", "#FF6347"),
                         legend.title = "Group")



# We can also used the circle representation, more visually.
# But for better understanding is better the number representation.
# corrplot::corrplot(mcor1, method = "circle")


# Save stuff -------------------------------------------------------------------
# 1 scenario - 1 file
write.csv(data1, "syntheticData/scenario1.csv", row.names=FALSE)
write.csv(data2, "syntheticData/scenario2.csv", row.names=FALSE)
write.csv(data3, "syntheticData/scenario3.csv", row.names=FALSE)
write.csv(data4, "syntheticData/scenario4.csv", row.names=FALSE)
write.csv(data5, "syntheticData/scenario5.csv", row.names=FALSE)

# Save both beta coefficients used
beta_data <- data.frame(
  original_beta=beta_coef,
  half_zero_beta=beta_coef_half_zeros
)
write.csv(beta_data, "syntheticData/beta_coefficients.csv", row.names=FALSE)

# Save the grouped plot
ggsave("syntheticData/PCA_plot.png", plot=pca_plot, width=10, height=8, dpi=300)
