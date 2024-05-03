################################################################################
## Project: Advanced Regression and Prediction
##
## Script for Option 1: Compare the performance of different regularization 
## procedures through a simulation study, constructing an error measure.
##
## Authors:
##  - Díaz-Plaza Cabrera, Miguel
##  - Levenfeld Sabau, Gabriela
################################################################################


# Load required libraries and files --------------------------------------------
library(glmnet)
library(caret)
library(MASS) # X matrix with predictors correlated

source("functions/split_data.R")

# Set up all parameters used ---------------------------------------------------
set.seed(1234) # for reproducibility
n <- 240 # nums of observations, must be divisible by 3! (for scenario 5)
p <- 10 # nums of predictors


# Synthetic data generation ----------------------------------------------------
# 1. Generate random values for the betas (as uniform integers in [-10,10])
beta_coef <- runif(p+1, min=-10, max=10) # Include beta_0 for the intercept

print('Values of the beta coefficients:')
for (i in 1:(p+1)){
  print(beta_coef[i])
}

# Scenario 1 ------------------------------------------------------------------
# 2. Generating Feature Matrix

# Each element is drawn independently:
# No correlation between variables, X
# Follows a Normal distribution (0, 1)
mu_X <- 0; sigma_X <- 1
X_0 <- matrix(rep(1, n), ncol=1)
X_1 <- matrix(rnorm(n*p, mean=mu_X, sd=sigma_X), ncol=p)
X <- cbind(X_0, X_1)

# 3. Creating Linear Relationship
y_lin <- X %*% beta_coef

# 4. Introducing Random Error (Noise) -> N(mean=0, sd=2)
normal_error <- stats::rnorm(n, mean=0, sd=2)
y <- y_lin + normal_error

# 5. Final Outputs
data1 <- data.frame(y=y, X)

# Get rid off the intercept, no need anymore
data1 <- subset(data1, select=-c(X1))


## Preparing the dataset --------------------------------------------------------

mcor <- cor(data1)
mcor
abs(mcor)>0.8
corrplot::corrplot(mcor, method = "number")
corrplot::corrplot(mcor, method = "circle")

# Another option for the correlation matrix
# library(GGally)
# ggpairs(data_without_intercept,
#         aes(fill="pink"),
#         lower = list(continuous = "points", combo = "box_no_facet"),
#         upper = list(continuous = "cor"),
#         diag = list(continuous = "barDiag"),
#         title = "Scatter plot matrix"
# )

aux <- split_data(data1)

y_train <- aux$train$y
y_test <- aux$test$y

x_train <- as.matrix(subset(aux$train, select = -c(y)))
x_test <- as.matrix(subset(aux$test, select = -c(y)))

## OLS ---------------------
ols <- lm(y ~ ., data = aux$train)

## Ridge regression ------------------
ridge_mod1 <- glmnet(x_train, y_train, alpha =0)
#ridge1 <- cv.glmnet(x=x, y=y, alpha = 0, lambda = lambdaGrid, nfolds = 10)

plot(ridge_mod1,  label = TRUE , xvar = "lambda")

# 10-fold cross-validation. Change the seed for a different result
kcvRidge <- cv.glmnet(x=x_train, y=y_train, alpha=0, nfolds=10)

## Lasso regression ------------------
lassoMod <- glmnet(x=x_train, y=y_train, alpha = 1)
plot(lassoMod, xvar = "lambda", label = TRUE)
modLassoCV <- kcvLasso$glmnet.fit
kcvLasso <- cv.glmnet(x=x_train, y=y_train, alpha= 1, nfolds=10)

## Elastic Net regression ------------------
system.time(elasticnet <- caret::train(x_train, y_train, method = "glmnet",
                                  preProc = c("zv", "center", "scale"),
                                  trControl = trainControl(method = "cv", number = 10),
                                  tuneLength = 10))
elasticnet


#Selects the LASSO
ggplot(elasticnet, highlight = TRUE)

## Predictions -----------------------------
predict(ridge_mod1, type = "coefficients")

newx=model.matrix(y~.,data=test)[,-1]

predRidge <- predict(ridge_mod1, newx = newx)#s=lambda.1SE
predLasso <- predict(kcvLasso, newx = newx)
predEnet <-predict(elasticnet,newx=newx)
predols <- predict(modOLSBIC, newdata =test)
#predBIC<- predict(ols, newdata = test)


accuracy <- function(pred, obs, na.rm = FALSE, 
                     tol = sqrt(.Machine$double.eps)) {
  err <- obs - pred     # Errores
  if(na.rm) {
    is.a <- !is.na(err)
    err <- err[is.a]
    obs <- obs[is.a]
  }  
  return(c(
    mep = mean(err),           # Mean error of prediction
    msep = mean(err^2), # Mean squared error of prediction
    maep = mean(abs(err))    # Mean absolute error of prediction
  ))
}


obs<-test$y
round(accuracy(predRidge, obs),2)
round(accuracy(predLasso, obs),2)
round(accuracy(predEnet, obs),2)
#round(accuracy(predBIC, obs),2)


# Scenario 2 ------------------------------------------------------------------
# Error/Noise distribution follows a Exponential distribution

# Matrix X and y: Same as in scenario 1
# betas coefficients are the same

# 4. Introducing Random Error (Noise) -> EXPONENTIAL
lamdba_param <- 0.35 
exp_error <- stats::rexp(n, rate=lamdba_param)
y <- y_lin + exp_error

# 5. Final Outputs
data2 <- data.frame(y=y, X)
# Get rid off the intercept, no need anymore
data2 <- subset(data2, select=-c(X1))

# Correlation between the predictors and the output change (not too much)
mcor2 <- cor(data2)
mcor2
abs(mcor2)>0.8
corrplot::corrplot(mcor2, method = "number")
corrplot::corrplot(mcor2, method = "circle")

aux2 <- split_data(data2)

y_train2 <- aux2$train$y
y_test2 <- aux2$test$y

x_train2 <- as.matrix(subset(aux2$train, select = -c(y)))
x_test2 <- as.matrix(subset(aux2$test, select = -c(y)))

## OLS ---------------------
ols2 <- lm(y ~ ., data = aux2$train)


# Scenario 3 ------------------------------------------------------------------
#https://medium.com/@marc.jacobs012/drawing-and-plotting-observations-from-a-multivariate-normal-distribution-using-r-4c2b2f64e1a3

# Same beta coefficients as before scenarios
# 2. Generating Feature Matrix

# Predictors are correlated
# Predictors follows a Normal distribution (0, 1)
rho <- 0.5 # Positive correlation coefficient
Sigma <- matrix(rho, nrow=p, ncol=p)
diag(Sigma) <- 1 # Set the diagonal to 1 (correlation with itself)
X_correlated <- mvrnorm(n, mu=rep(0, p), Sigma=Sigma)
# TODO: Create negative correlation on the data

# Add the intercept column
#X_0 <- matrix(rep(1, n), ncol=1) # Same as before
X <- cbind(X_0, X_correlated)

# 3. Creating Linear Relationship
y_lin <- X %*% beta_coef

# 4. Introducing Random Error (Noise) -> N(mean=0, sd=2)
y <- y_lin + normal_error

# 5. Final Outputs
data3 <- data.frame(y=y, X)

# Get rid off the intercept, no need anymore
data3 <- subset(data3, select=-c(X1))

## Preparing the dataset --------------------------------------------------------

mcor3 <- cor(data3)
mcor3
abs(mcor3)>0.8
corrplot::corrplot(mcor3, method = "number")
corrplot::corrplot(mcor3, method = "circle")

aux3 <- split_data(data3)

y_train3 <- aux3$train$y
y_test3 <- aux3$test$y

x_train3 <- as.matrix(subset(aux3$train, select = -c(y)))
x_test3 <- as.matrix(subset(aux3$test, select = -c(y)))

## OLS ---------------------
ols3 <- lm(y ~ ., data = aux3$train)


# Scenario 4 ------------------------------------------------------------------
# Half of the beta coefficients are equal to zero

zero_index <- c(2, 3, 8, 9, 10)
beta_coef_half_zeros <- beta_coef
beta_coef_half_zeros[zero_index] <- 0

# Predictors X are correlated
# Error/Noise distribution follows a normal distribution (same as before)

# 2. Generating Feature Matrix
rho <- 0.5 # Positive correlation coefficient
Sigma <- matrix(rho, nrow=p, ncol=p)
diag(Sigma) <- 1 # Set the diagonal to 1 (correlation with itself)
X_correlated <- mvrnorm(n, mu=rep(0, p), Sigma=Sigma)

# Add the intercept column
#X_0 <- matrix(rep(1, n), ncol=1) # Same as before
X <- cbind(X_0, X_correlated)

# 3. Creating Linear Relationship
y_lin <- X %*% beta_coef_half_zeros

# 4. Introducing Random Error (Noise) -> N(mean=0, sd=2)
y <- y_lin + normal_error

# 5. Final Outputs
data4 <- data.frame(y=y, X)

# Get rid off the intercept, no need anymore
data4 <- subset(data4, select=-c(X1))

## Preparing the dataset --------------------------------------------------------

mcor4 <- cor(data4)
mcor4
abs(mcor4)>0.8
corrplot::corrplot(mcor4, method = "number")
corrplot::corrplot(mcor4, method = "circle")

aux4 <- split_data(data4)

y_train4 <- aux4$train$y
y_test4 <- aux4$test$y

x_train4 <- as.matrix(subset(aux4$train, select = -c(y)))
x_test4 <- as.matrix(subset(aux4$test, select = -c(y)))

## OLS ---------------------
ols4 <- lm(y ~ ., data = aux4$train)

# Scenario 5 ------------------------------------------------------------------
# Grouped observations
# https://watermark.silverchair.com/jrsssb_67_2_301.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAA2MwggNfBgkqhkiG9w0BBwagggNQMIIDTAIBADCCA0UGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMHvbXDzKU7FNlJ1v3AgEQgIIDFgsu4ebwEuKZpsHFmC3KIVjnHYHlD32OpF8svtc5UrPVwBAcBRkAIsgi579GOK_bnkVRnH5rhT-rWHycJYRXSnCndspVTaZnDJBJUJto1WMyVgOtO7-3rCViTjxlg2ZMd9z67sP3v1nsr6A29Gk2Jsl3Psq9TXkR0osrpqgVTSYwAEaUNBCI8nMRt2F_ZCTZ_mg-LRDQiM6AiBmKg5x3ZjH2meymIjC3pYgWGT7swTeAuhuznZPXVs9n0AhaD7C0mIt-22RPgiR2Wd6l2IQwq-HXC_yUQxyPerr6q0UAM5jWZIZzajashvr3X0gHow_PRldGzo8zEJOuqJnG5SRsS_FEAB_SSsyf1u3HPL9HEjk7iVBFP7kmmIsHDdKTF-zy47iTyXhnrboHJ271mIA5PgvW05X0pkjvIoN81YbMe07bwNEhVcDnQPGXkNT8yZcBhMQIjuqEB_ZtsP9mcwZpjKPU8J8kSg2S0-RZagx6IB2KHb7C01ae_pw2jDV13VUQTRUR6TasE9m67G0Vy5273bPH9ByNrWXRaMMoW9-vA1YnC_LaXhePSgQ78v-qpTockMVUj6wVE3yCNcgaaNLXmmcXPgy-s7szRr1u61csuoHEOSRHKaOCLHhehf168p9k3vIcGSPD36QP4FFMvYzLF1sBa9yDHwDsTIAbjrWNB4XGtzTZbk5fNQvVCltitNZR-ceCCeNM4ZfFrEMtPd2C9sdWLSJeYU5eQszdZMetPYLe5wbAzjcH10DN8v5pG0dvijgeVqD0WdfmMmPSFAm6hOJd7yU8M21ORZaPRUUZpqp-_YfVnRrwFM9v9wfbCHpbZqxbtkB5MdN0OX2EhgyLuQFFhvUBjzTbO1y-ydVpLEnUolJIKNHtrcE_dwpD2_IV4u88xXoLshSHNjUHBwdItflhDbpVWiFY5A1BXjRD7_hmn0qfIhqt3-TO2AKdu3Nucv9nwpKksPqVZZHeapC-3LWK4RTLy6c1QWxyv1KHywrIeGIjnBjVMZAoVbJDGye0nd5osfphS1F9XhfWXJjTsI2JP7aP5Yo

num_groups <- 3
obs_per_group <- n/num_groups # For creating 3 groups -> 80 obvs per group

# Means for each group in order to simulate groups
group_means <- matrix(c(rep(-1.5,p), rep(1,p), rep(3,p)), ncol=p, byrow=TRUE)
group_data <- vector("list", num_groups)

# Generate data for each group
for (i in 1:num_groups) {
  # Generate feature matrix with group-specific means
  Sigma <- diag(p)
  X_group <- mvrnorm(obs_per_group, mu=group_means[i, ], Sigma=Sigma)
  
  # Add the intercept
  X_0_group <- matrix(rep(1, obs_per_group), ncol=1)
  X_group <- cbind(X_0_group, X_group)
  
  y_lin_group <- X_group %*% beta_coef
  
  # Introduce normal error -> N(mean=0, sd=2)
  normal_error_group <- rnorm(obs_per_group, mean=0, sd=2)
  y_group <- y_lin_group + normal_error_group
  
  # Store in group_data
  group_data[[i]] <- data.frame(group=factor(i), y=y_group, X=X_group)
}
# Combine group data into one data frame
data5 <- do.call(rbind, group_data)
# Get rid off the intercept, no need anymore
pca_data <- data5
pca_data <- subset(data5, select=-c(X.1))
data5 <- subset(data5, select=-c(X.1, group))

## Extra stuff: cheking the group creation --------------------
library(factoextra) # For PCA and its visualization

# PCA: excluding group (1) and y (2) variables
pca_2_vars <- prcomp(pca_data[, -c(1,2)], scale.=TRUE)

# Check groups by biplot PCA
fviz_pca_ind(pca_2_vars,
             col.ind = pca_data$group, # Color by group
             palette = c("#4682B4", "#32CD32", "#FF6347"),
             legend.title = "Group")


## Preparing the dataset --------------------------------------------------------

mcor5 <- cor(data5)
mcor5
abs(mcor5)>0.8
corrplot::corrplot(mcor5, method = "number")
corrplot::corrplot(mcor5, method = "circle")

aux5 <- split_data(data5)

y_train5 <- aux5$train$y
y_test5 <- aux5$test$y

x_train5 <- as.matrix(subset(aux5$train, select = -c(y)))
x_test5 <- as.matrix(subset(aux5$test, select = -c(y)))

## OLS ---------------------
ols5 <- lm(y ~ ., data = aux5$train)
