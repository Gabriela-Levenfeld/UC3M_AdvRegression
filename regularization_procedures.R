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



# Scenario 3 -------------------------------------------------------------------
simulations3 <- read.csv("syntheticData/scenario3.csv")
aux3 <- split_data(simulations3)

x_train3 <- as.matrix(subset(aux3$train, select=-c(y)))
x_test3 <- as.matrix(subset(aux3$test, select=-c(y)))

y_train3 <- aux3$train$y
y_test3 <- aux3$test$y

## Regularization techniques ----------------------------------------------------
# One of this models might be used as a benchmark
ols3 <- lm(y~., data = aux3$train) # OLS
olsBIC_mod3 <- stepAIC(ols3, k=log(nrow(x_train3)), trace=0) # OLS combined with the BIC


# Ridge regression -> alpha=0
ridge_mod3 <- glmnet(x_train3, y_train3, alpha=0)

# 10-fold cross-validation
kcvRidge_mod3 <- cv.glmnet(x=x_train3, y=y_train3, alpha=0, nfolds=10)
kcvRidge_mod3$lambda.min # 0.848094
length(kcvRidge_mod3$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod3 <- 10^seq(log10(kcvRidge_mod3$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod3 <- cv.glmnet(x=x_train3, y=y_train3, alpha=0, nfolds=10, lambda=lambdaGrid1_mod3)
kcvRidge2_mod3$lambda.min # 0.1
min(kcvRidge2_mod3$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod3$lambda.1se # 0.3938968

finalRidgeCV_mod3 <- kcvRidge2_mod3$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod3, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod3$lambda.min, kcvRidge2_mod3$lambda.1se)))

# Predictions using the lambda.1se
pred_Ridge3_coef <- predict(finalRidgeCV_mod3, type="coefficients", s=kcvRidge2_mod3$lambda.1se)
pred_Ridge3_y <- predict(finalRidgeCV_mod3, newx=x_test3, s=kcvRidge2_mod3$lambda.1se)

# REVIEW: Should we compute the error on the beta coeff?


# Lasso regression -> alpha=1
lasso_mod3 <- glmnet(x=x_train3, y=y_train3, alpha=1)
kcvLasso_mod3 <- cv.glmnet(x=x_train3, y=y_train3, alpha=1, nfolds=10)
kcvLasso_mod3$lambda.min
length(kcvLasso_mod3$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod3 <- 10^seq(log10(kcvLasso_mod3$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod3 <- cv.glmnet(x=x_train3, y=y_train3, alpha=1, nfolds=10, lambda=lambdaGrid2_mod3)
kcvLasso2_mod3$lambda.min # 0.1
min(kcvLasso2_mod3$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod3$lambda.1se # 0.1709883

finalLassoCV_mod3 <- kcvLasso2_mod3$glmnet.fit

# Predictions using the lambda.1se
pred_Lasso3_coef <- predict(finalLassoCV_mod3, type="coefficients", s=kcvLasso2_mod3$lambda.1se)
pred_Lasso3_y <- predict(finalLassoCV_mod3, newx=x_test3, s=kcvLasso2_mod3$lambda.1se)

predict(kcvLasso2_mod3, type = "coefficients",
        s = c(kcvLasso2_mod3$lambda.min, kcvLasso2_mod3$lambda.1se))
# REVIEW. There is no difference between values... weird!


# Elastic Net regression
elasticNet_mod3 <- caret::train(x_train3, y_train3, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod3, highlight = TRUE)

elasticNet_mod3$bestTune$lambda # 0.04830804
elasticNet_mod3$bestTune$alpha # 0.1

kcvEnet_mod3 <- cv.glmnet(x=x_train3, y=y_train3, alpha=0.1, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod3 <- kcvEnet_mod3$glmnet.fit

# Predictions
pred_elasticNet3_coef <- predict(finalEnetCV_mod3, s=kcvEnet_mod3$lambda.1se, type="coefficients")
pred_elasticNet3_y <- predict(finalEnetCV_mod3, newx=x_test3, s=kcvEnet_mod3$lambda.1se)


## Errors on test data ---------------------------------------------------------
# Summary with the errors of three models
round(error_metrics(pred_Ridge3_y, y_test3), 2)
round(error_metrics(pred_Lasso3_y, y_test3), 2)
round(error_metrics(pred_elasticNet3_y, y_test3), 2)



# Scenario 4 -------------------------------------------------------------------
simulations4 <- read.csv("syntheticData/scenario4.csv")
aux4 <- split_data(simulations4)

x_train4 <- as.matrix(subset(aux4$train, select=-c(y)))
x_test4 <- as.matrix(subset(aux4$test, select=-c(y)))

y_train4 <- aux4$train$y
y_test4 <- aux4$test$y

## Regularization techniques ----------------------------------------------------
# One of this models might be used as a benchmark
ols4 <- lm(y~., data = aux4$train) # OLS
olsBIC_mod4 <- stepAIC(ols4, k=log(nrow(x_train4)), trace=0) # OLS combined with the BIC


# Ridge regression -> alpha=0
ridge_mod4 <- glmnet(x_train4, y_train4, alpha=0)

# 10-fold cross-validation
kcvRidge_mod4 <- cv.glmnet(x=x_train4, y=y_train4, alpha=0, nfolds=10)
kcvRidge_mod4$lambda.min # 0.6629677
length(kcvRidge_mod4$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod4 <- 10^seq(log10(kcvRidge_mod4$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod4 <- cv.glmnet(x=x_train4, y=y_train4, alpha=0, nfolds=10, lambda=lambdaGrid1_mod4)
kcvRidge2_mod4$lambda.min # 0.1
min(kcvRidge2_mod4$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod4$lambda.1se # 0.4119279

finalRidgeCV_mod4 <- kcvRidge2_mod4$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod4, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod4$lambda.min, kcvRidge2_mod4$lambda.1se)))

# Predictions using the lambda.1se
pred_Ridge4_coef <- predict(finalRidgeCV_mod4, type="coefficients", s=kcvRidge2_mod4$lambda.1se)
pred_Ridge4_y <- predict(finalRidgeCV_mod4, newx=x_test4, s=kcvRidge2_mod4$lambda.1se)

# REVIEW: Should we compute the error on the beta coeff?


# Lasso regression -> alpha=1
lasso_mod4 <- glmnet(x=x_train4, y=y_train4, alpha=1)
kcvLasso_mod4 <- cv.glmnet(x=x_train4, y=y_train4, alpha=1, nfolds=10)
kcvLasso_mod4$lambda.min
length(kcvLasso_mod4$lambda)
# 66, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod4 <- 10^seq(log10(kcvLasso_mod4$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod4 <- cv.glmnet(x=x_train4, y=y_train4, alpha=1, nfolds=10, lambda=lambdaGrid2_mod4)
kcvLasso2_mod4$lambda.min # 0.1
min(kcvLasso2_mod4$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod4$lambda.1se # 0.2604009

finalLassoCV_mod4 <- kcvLasso2_mod4$glmnet.fit

# Predictions using the lambda.1se
pred_Lasso4_coef <- predict(finalLassoCV_mod4, type="coefficients", s=kcvLasso2_mod4$lambda.1se)
pred_Lasso4_y <- predict(finalLassoCV_mod4, newx=x_test4, s=kcvLasso2_mod4$lambda.1se)

predict(kcvLasso2_mod4, type = "coefficients",
        s = c(kcvLasso2_mod4$lambda.min, kcvLasso2_mod4$lambda.1se))
# There is difference!!


# Elastic Net regression
elasticNet_mod4 <- caret::train(x_train4, y_train4, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod4, highlight = TRUE)

elasticNet_mod4$bestTune$lambda # 0.08723769
elasticNet_mod4$bestTune$alpha # 0.1

kcvEnet_mod4 <- cv.glmnet(x=x_train4, y=y_train4, alpha=0.1, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod4 <- kcvEnet_mod4$glmnet.fit

# Predictions
pred_elasticNet4_coef <- predict(finalEnetCV_mod4, s=kcvEnet_mod4$lambda.1se, type="coefficients")
pred_elasticNet4_y <- predict(finalEnetCV_mod4, newx=x_test4, s=kcvEnet_mod4$lambda.1se)


## Errors on test data ---------------------------------------------------------
# Summary with the errors of three models
round(error_metrics(pred_Ridge4_y, y_test4), 2)
round(error_metrics(pred_Lasso4_y, y_test4), 2)
round(error_metrics(pred_elasticNet4_y, y_test4), 2)



# Scenario 5 -------------------------------------------------------------------
simulations5 <- read.csv("syntheticData/scenario5.csv")
aux5 <- split_data(simulations5)

x_train5 <- as.matrix(subset(aux5$train, select=-c(y)))
x_test5 <- as.matrix(subset(aux5$test, select=-c(y)))

y_train5 <- aux5$train$y
y_test5 <- aux5$test$y

## Regularization techniques ----------------------------------------------------
# One of this models might be used as a benchmark
ols5 <- lm(y~., data = aux5$train) # OLS
olsBIC_mod5 <- stepAIC(ols5, k=log(nrow(x_train5)), trace=0) # OLS combined with the BIC


# Ridge regression -> alpha=0
ridge_mod5 <- glmnet(x_train5, y_train5, alpha=0)

# 10-fold cross-validation
kcvRidge_mod5 <- cv.glmnet(x=x_train5, y=y_train5, alpha=0, nfolds=10)
kcvRidge_mod5$lambda.min # 1.84889
length(kcvRidge_mod5$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod5 <- 10^seq(log10(kcvRidge_mod5$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod5 <- cv.glmnet(x=x_train5, y=y_train5, alpha=0, nfolds=10, lambda=lambdaGrid1_mod5)
kcvRidge2_mod5$lambda.min # 0.1
min(kcvRidge2_mod5$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod5$lambda.1se # 0.2080346

finalRidgeCV_mod5 <- kcvRidge2_mod5$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod5, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod5$lambda.min, kcvRidge2_mod5$lambda.1se)))

# Predictions using the lambda.1se
pred_Ridge5_coef <- predict(finalRidgeCV_mod5, type="coefficients", s=kcvRidge2_mod5$lambda.1se)
pred_Ridge5_y <- predict(finalRidgeCV_mod5, newx=x_test5, s=kcvRidge2_mod5$lambda.1se)

# REVIEW: Should we compute the error on the beta coeff?


# Lasso regression -> alpha=1
lasso_mod5 <- glmnet(x=x_train5, y=y_train5, alpha=1)
kcvLasso_mod5 <- cv.glmnet(x=x_train5, y=y_train5, alpha=1, nfolds=10)
kcvLasso_mod5$lambda.min
length(kcvLasso_mod5$lambda)
# 69, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod5 <- 10^seq(log10(kcvLasso_mod5$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod5 <- cv.glmnet(x=x_train5, y=y_train5, alpha=1, nfolds=10, lambda=lambdaGrid2_mod5)
kcvLasso2_mod5$lambda.min # 0.1
min(kcvLasso2_mod5$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod5$lambda.1se # 0.1576828

finalLassoCV_mod5 <- kcvLasso2_mod5$glmnet.fit

# Predictions using the lambda.1se
pred_Lasso5_coef <- predict(finalLassoCV_mod5, type="coefficients", s=kcvLasso2_mod5$lambda.1se)
pred_Lasso5_y <- predict(finalLassoCV_mod5, newx=x_test5, s=kcvLasso2_mod5$lambda.1se)

predict(kcvLasso2_mod5, type = "coefficients",
        s = c(kcvLasso2_mod5$lambda.min, kcvLasso2_mod5$lambda.1se))
# There is difference!!


# Elastic Net regression
elasticNet_mod5 <- caret::train(x_train5, y_train5, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod5, highlight = TRUE)

elasticNet_mod5$bestTune$lambda # 0.01973394
elasticNet_mod5$bestTune$alpha # 0.2

kcvEnet_mod5 <- cv.glmnet(x=x_train5, y=y_train5, alpha=0.3, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod5 <- kcvEnet_mod5$glmnet.fit

# Predictions
pred_elasticNet5_coef <- predict(finalEnetCV_mod5, s=kcvEnet_mod5$lambda.1se, type="coefficients")
pred_elasticNet5_y <- predict(finalEnetCV_mod5, newx=x_test5, s=kcvEnet_mod5$lambda.1se)


## Errors on test data ---------------------------------------------------------
# Summary with the errors of three models
round(error_metrics(pred_Ridge5_y, y_test5), 2)
round(error_metrics(pred_Lasso5_y, y_test5), 2)
round(error_metrics(pred_elasticNet5_y, y_test5), 2)
