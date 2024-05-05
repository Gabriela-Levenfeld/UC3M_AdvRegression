################################################################################
## Project: Advanced Regression and Prediction
##
## Script for Option 1: Compare the performance of different regularization 
## procedures through a simulation study, constructing an error measure.
##
################################################################################

# Load libraries and functions -------------------------------------------------
library(glmnet)
library(caret)
library(ggplot2)
source("functions/split_data.R")
source("functions/error_metrics.R")


set.seed(1234) # for reproducibility
# Scenario 1 -------------------------------------------------------------------
simulations1 <- read.csv("syntheticData/scenario1.csv")
aux1 <- split_data(simulations1)

x_train <- as.matrix(subset(aux1$train, select=-c(y)))
x_test <- as.matrix(subset(aux1$test, select=-c(y)))

y_train <- aux1$train$y
y_test <- aux1$test$y

## Regularization techniques ----------------------------------------------------
# One of this models might be used as a benchmark
ols1 <- lm(y~., data = aux1$train) # OLS
olsBIC_mod1 <- stepAIC(ols1, k=log(nrow(x_train)), trace=0) # OLS combined with the BIC


# Ridge regression -> alpha=0
ridge_mod1 <- glmnet(x_train, y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod1 <- cv.glmnet(x=x_train, y=y_train, alpha=0, nfolds=10)
# kcvRidge_mod1$lambda.min -> 0.9826949
# length(kcvRidge_mod1$lambda) -> 100 which is the last CV position
# Potential problem! Minimum occurs at one extreme of the lambda grid in which
# CV is done. The grid was automatically selected, but can be manually inputted.
# To solve this issue, we extend the range

lambdaGrid1_mod1 <- 10^seq(log10(kcvRidge_mod1$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod1 <- cv.glmnet(x=x_train, y=y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod1)
kcvRidge2_mod1$lambda.min # 0.1
min(kcvRidge2_mod1$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod1$lambda.1se # 0.5897145

finalRidgeCV_mod1 <- kcvRidge2_mod1$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod1, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod1$lambda.min, kcvRidge2_mod1$lambda.1se)))

# Predictions using the lambda.1se
pred_Ridge1_coef <- predict(finalRidgeCV_mod1, type="coefficients", s=kcvRidge2_mod1$lambda.1se)
pred_Ridge1_y <- predict(finalRidgeCV_mod1, newx=x_test, s=kcvRidge2_mod1$lambda.1se)

# REVIEW: Should we compute the error on the beta coeff?


# Lasso regression -> alpha=1
lasso_mod1 <- glmnet(x=x_train, y=y_train, alpha=1)
kcvLasso_mod1 <- cv.glmnet(x=x_train, y=y_train, alpha=1, nfolds=10)
kcvLasso_mod1$lambda.min
length(kcvLasso_mod1$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod1 <- 10^seq(log10(kcvLasso_mod1$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod1 <- cv.glmnet(x=x_train, y=y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod1)
kcvLasso2_mod1$lambda.min # 0.1
min(kcvLasso2_mod1$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod1$lambda.1se # 0.1687808

finalLassoCV_mod1 <- kcvLasso2_mod1$glmnet.fit

# Predictions using the lambda.1se
pred_Lasso1_coef <- predict(finalLassoCV_mod1, type="coefficients", s=kcvLasso2_mod1$lambda.1se)
pred_Lasso1_y <- predict(finalLassoCV_mod1, newx=x_test, s=kcvLasso2_mod1$lambda.1se)

predict(kcvLasso2_mod1, type = "coefficients",
        s = c(kcvLasso2_mod1$lambda.min, kcvLasso2_mod1$lambda.1se))
# REVIEW. There is no difference between values... weird!


# Elastic Net regression
elasticNet_mod1 <- caret::train(x_train, y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod1, highlight = TRUE)

elasticNet_mod1$bestTune$lambda # 0.02423024
elasticNet_mod1$bestTune$alpha # 0.6

kcvEnet_mod1 <- cv.glmnet(x=x_train, y=y_train, alpha=0.6, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod1 <- kcvEnet_mod1$glmnet.fit

# Predictions
pred_elasticNet1_coef <- predict(finalEnetCV_mod1, s=kcvEnet_mod1$lambda.1se, type="coefficients")
pred_elasticNet1_y <- predict(finalEnetCV_mod1, newx=x_test, s=kcvEnet_mod1$lambda.1se)


## Errors on test data ---------------------------------------------------------
# Summary with the errors of three models
round(error_metrics(pred_Ridge1_y, y_test), 2)
round(error_metrics(pred_Lasso1_y, y_test), 2)
round(error_metrics(pred_elasticNet1_y, y_test), 2)



# Scenario 2 -------------------------------------------------------------------
simulations2 <- read.csv("syntheticData/scenario2.csv")
aux2 <- split_data(simulations2)

x_train2 <- as.matrix(subset(aux2$train, select=-c(y)))
x_test2 <- as.matrix(subset(aux2$test, select=-c(y)))

y_train2 <- aux2$train$y
y_test2 <- aux2$test$y

## Regularization techniques ----------------------------------------------------
# One of this models might be used as a benchmark
ols2 <- lm(y~., data = aux2$train) # OLS
olsBIC_mod2 <- stepAIC(ols2, k=log(nrow(x_train2)), trace=0) # OLS combined with the BIC


# Ridge regression -> alpha=0
ridge_mod2 <- glmnet(x_train2, y_train2, alpha=0)

# 10-fold cross-validation
kcvRidge_mod2 <- cv.glmnet(x=x_train2, y=y_train2, alpha=0, nfolds=10)
kcvRidge_mod2$lambda.min # 1.099348
length(kcvRidge_mod2$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod2 <- 10^seq(log10(kcvRidge_mod2$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod2 <- cv.glmnet(x=x_train2, y=y_train2, alpha=0, nfolds=10, lambda=lambdaGrid1_mod2)
kcvRidge2_mod2$lambda.min # 0.1
min(kcvRidge2_mod2$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod2$lambda.1se # 1.093534

finalRidgeCV_mod2 <- kcvRidge2_mod2$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod2, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod2$lambda.min, kcvRidge2_mod2$lambda.1se)))

# Predictions using the lambda.1se
pred_Ridge2_coef <- predict(finalRidgeCV_mod2, type="coefficients", s=kcvRidge2_mod2$lambda.1se)
pred_Ridge2_y <- predict(finalRidgeCV_mod2, newx=x_test2, s=kcvRidge2_mod2$lambda.1se)


# Lasso regression -> alpha=1
lasso_mod2 <- glmnet(x=x_train2, y=y_train2, alpha=1)
kcvLasso_mod2 <- cv.glmnet(x=x_train2, y=y_train2, alpha=1, nfolds=10)
kcvLasso_mod2$lambda.min
length(kcvLasso_mod2$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod2 <- 10^seq(log10(kcvLasso_mod2$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod2 <- cv.glmnet(x=x_train2, y=y_train2, alpha=1, nfolds=10, lambda=lambdaGrid2_mod2)
kcvLasso2_mod2$lambda.min # 0.1
min(kcvLasso2_mod2$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod2$lambda.1se # 0.3324171

finalLassoCV_mod2 <- kcvLasso2_mod2$glmnet.fit

# Predictions using the lambda.1se
pred_Lasso2_coef <- predict(finalLassoCV_mod2, type="coefficients", s=kcvLasso2_mod2$lambda.1se)
pred_Lasso2_y <- predict(finalLassoCV_mod2, newx=x_test2, s=kcvLasso2_mod2$lambda.1se)

predict(kcvLasso2_mod2, type = "coefficients",
        s=c(kcvLasso2_mod2$lambda.min, kcvLasso2_mod2$lambda.1se))
# It deactivate the X10 variable -> NICE!!


# Elastic Net regression
elasticNet_mod2 <- caret::train(x_train2, y_train2, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod2, highlight = TRUE)

elasticNet_mod2$bestTune$lambda # 0.02426817
elasticNet_mod2$bestTune$alpha # 0.5

kcvEnet_mod2 <- cv.glmnet(x=x_train2, y=y_train2, alpha=0.5, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod2 <- kcvEnet_mod2$glmnet.fit

# Predictions
pred_elasticNet2_coef <- predict(finalEnetCV_mod2, s=kcvEnet_mod2$lambda.1se, type="coefficients")
pred_elasticNet2_y <- predict(finalEnetCV_mod2, newx=x_test2, s=kcvEnet_mod2$lambda.1se)


## Errors on test data ---------------------------------------------------------
# Summary with the errors of three models
round(error_metrics(pred_Ridge2_y, y_test2), 2)
round(error_metrics(pred_Lasso2_y, y_test2), 2)
round(error_metrics(pred_elasticNet2_y, y_test2), 2)
