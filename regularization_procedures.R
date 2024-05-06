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
library(ggplot2)
source("functions/split_data.R")
source("functions/summary_mod_performance.R")


set.seed(1234) # for reproducibility
# Scenario 1 -------------------------------------------------------------------
simulations1 <- read.csv("syntheticData/scenario1.csv")
aux1 <- split_data(simulations1)

## Benchmark -------------------------------------------------------------------
ols1 <- lm(y~., data = aux1$train) # OLS
olsBIC_mod1 <- stepAIC(ols1, k=log(nrow(aux1$x_train)), trace=0) # OLS combined with the BIC


## Ridge (alpha=0) -------------------------------------------------------------
ridge_mod1 <- glmnet(aux1$x_train, aux1$y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod1 <- cv.glmnet(x=aux1$x_train, y=aux1$y_train, alpha=0, nfolds=10)
# kcvRidge_mod1$lambda.min -> 0.9826949
# length(kcvRidge_mod1$lambda) -> 100 which is the last CV position
# Potential problem! Minimum occurs at one extreme of the lambda grid in which
# CV is done. The grid was automatically selected, but can be manually inputted.
# To solve this issue, we extend the range

lambdaGrid1_mod1 <- 10^seq(log10(kcvRidge_mod1$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod1 <- cv.glmnet(x=aux1$x_train, y=aux1$y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod1)
kcvRidge2_mod1$lambda.min # 0.1
min(kcvRidge2_mod1$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod1$lambda.1se # 0.5897145

finalRidgeCV_mod1 <- kcvRidge2_mod1$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod1, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod1$lambda.min, kcvRidge2_mod1$lambda.1se)))

# Predictions using the lambda.1se
ridge_res_mod1 <- summary_mod_performance(aux1$x_test, aux1$y_test, kcvRidge2_mod1)


## Lasso (alpha=1) -------------------------------------------------------------
lasso_mod1 <- glmnet(x=aux1$x_train, y=aux1$y_train, alpha=1)
kcvLasso_mod1 <- cv.glmnet(x=aux1$x_train, y=aux1$y_train, alpha=1, nfolds=10)
kcvLasso_mod1$lambda.min
length(kcvLasso_mod1$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod1 <- 10^seq(log10(kcvLasso_mod1$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod1 <- cv.glmnet(x=aux1$x_train, y=aux1$y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod1)
kcvLasso2_mod1$lambda.min # 0.1
min(kcvLasso2_mod1$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod1$lambda.1se # 0.1687808

finalLassoCV_mod1 <- kcvLasso2_mod1$glmnet.fit

# Predictions using the lambda.1se
lasso_res_mod1 <- summary_mod_performance(aux1$x_test, aux1$y_test, kcvLasso2_mod1)

predict(kcvLasso2_mod1, type = "coefficients",
        s = c(kcvLasso2_mod1$lambda.min, kcvLasso2_mod1$lambda.1se))
# REVIEW. There is no difference between values... weird!


## Elastic Net ------------------------------------------------------------------
elasticNet_mod1 <- caret::train(aux1$x_train, aux1$y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod1, highlight = TRUE)

elasticNet_mod1$bestTune$lambda # 0.02423024
elasticNet_mod1$bestTune$alpha # 0.6

kcvEnet_mod1 <- cv.glmnet(x=aux1$x_train, y=aux1$y_train, alpha=0.6, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod1 <- kcvEnet_mod1$glmnet.fit

# Predictions using the lambda.1se
elasticNet_res_mod1 <- summary_mod_performance(aux1$x_test, aux1$y_test, kcvEnet_mod1)



# Scenario 2 -------------------------------------------------------------------
simulations2 <- read.csv("syntheticData/scenario2.csv")
aux2 <- split_data(simulations2)

## Benchmark -------------------------------------------------------------------
ols2 <- lm(y~., data = aux2$train) # OLS
olsBIC_mod2 <- stepAIC(ols2, k=log(nrow(aux2$x_train)), trace=0) # OLS combined with the BIC


## Ridge (alpha=0) -------------------------------------------------------------
ridge_mod2 <- glmnet(aux2$x_train, aux2$y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod2 <- cv.glmnet(x=aux2$x_train, y=aux2$y_train, alpha=0, nfolds=10)
kcvRidge_mod2$lambda.min # 1.099348
length(kcvRidge_mod2$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod2 <- 10^seq(log10(kcvRidge_mod2$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod2 <- cv.glmnet(x=aux2$x_train, y=aux2$y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod2)
kcvRidge2_mod2$lambda.min # 0.1
min(kcvRidge2_mod2$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod2$lambda.1se # 1.093534

finalRidgeCV_mod2 <- kcvRidge2_mod2$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod2, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod2$lambda.min, kcvRidge2_mod2$lambda.1se)))

# Predictions using the lambda.1se
ridge_res_mod2 <- summary_mod_performance(aux2$x_test, aux2$y_test, kcvRidge2_mod2)


## Lasso (alpha=1) -------------------------------------------------------------
lasso_mod2 <- glmnet(x=aux2$x_train, y=aux2$y_train, alpha=1)
kcvLasso_mod2 <- cv.glmnet(x=aux2$x_train, y=aux2$y_train, alpha=1, nfolds=10)
kcvLasso_mod2$lambda.min
length(kcvLasso_mod2$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod2 <- 10^seq(log10(kcvLasso_mod2$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod2 <- cv.glmnet(x=aux2$x_train, y=aux2$y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod2)
kcvLasso2_mod2$lambda.min # 0.1
min(kcvLasso2_mod2$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod2$lambda.1se # 0.3324171

finalLassoCV_mod2 <- kcvLasso2_mod2$glmnet.fit

# Predictions using the lambda.1se
lasso_res_mod2 <- summary_mod_performance(aux2$x_test, aux2$y_test, kcvLasso2_mod2)
predict(kcvLasso2_mod2, type = "coefficients",
        s=c(kcvLasso2_mod2$lambda.min, kcvLasso2_mod2$lambda.1se))
# It deactivate the X10 variable -> NICE!!


## Elastic Net ------------------------------------------------------------------
elasticNet_mod2 <- caret::train(aux2$x_train, aux2$y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod2, highlight = TRUE)

elasticNet_mod2$bestTune$lambda # 0.02426817
elasticNet_mod2$bestTune$alpha # 0.5

kcvEnet_mod2 <- cv.glmnet(x=aux2$x_train, y=aux2$y_train, alpha=0.5, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod2 <- kcvEnet_mod2$glmnet.fit

# Predictions using the lambda.1se
elasticNet_res_mod2 <- summary_mod_performance(aux2$x_test, aux2$y_test, kcvEnet_mod2)



# Scenario 3 -------------------------------------------------------------------
simulations3 <- read.csv("syntheticData/scenario3.csv")
aux3 <- split_data(simulations3)

## Benchmark -------------------------------------------------------------------
ols3 <- lm(y~., data = aux3$train) # OLS
olsBIC_mod3 <- stepAIC(ols3, k=log(nrow(aux3$x_train)), trace=0) # OLS combined with the BIC


## Ridge (alpha=0) -------------------------------------------------------------
ridge_mod3 <- glmnet(aux3$x_train, aux3$y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod3 <- cv.glmnet(x=aux3$x_train, y=aux3$y_train, alpha=0, nfolds=10)
kcvRidge_mod3$lambda.min # 0.848094
length(kcvRidge_mod3$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod3 <- 10^seq(log10(kcvRidge_mod3$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod3 <- cv.glmnet(x=aux3$x_train, y=aux3$y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod3)
kcvRidge2_mod3$lambda.min # 0.1
min(kcvRidge2_mod3$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod3$lambda.1se # 0.3938968

finalRidgeCV_mod3 <- kcvRidge2_mod3$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod3, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod3$lambda.min, kcvRidge2_mod3$lambda.1se)))

# Predictions using the lambda.1se
ridge_res_mod3 <- summary_mod_performance(aux3$x_test, aux3$y_test, kcvRidge2_mod3)


## Lasso (alpha=1) -------------------------------------------------------------
lasso_mod3 <- glmnet(x=aux3$x_train, y=aux3$y_train, alpha=1)
kcvLasso_mod3 <- cv.glmnet(x=aux3$x_train, y=aux3$y_train, alpha=1, nfolds=10)
kcvLasso_mod3$lambda.min
length(kcvLasso_mod3$lambda)
# 63, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod3 <- 10^seq(log10(kcvLasso_mod3$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod3 <- cv.glmnet(x=aux3$x_train, y=aux3$y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod3)
kcvLasso2_mod3$lambda.min # 0.1
min(kcvLasso2_mod3$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod3$lambda.1se # 0.1709883

finalLassoCV_mod3 <- kcvLasso2_mod3$glmnet.fit

# Predictions using the lambda.1se
lasso_res_mod3 <- summary_mod_performance(aux3$x_test, aux3$y_test, kcvLasso2_mod3)
predict(kcvLasso2_mod3, type = "coefficients",
        s = c(kcvLasso2_mod3$lambda.min, kcvLasso2_mod3$lambda.1se))
# REVIEW. There is no difference between values... weird!


## Elastic Net ------------------------------------------------------------------
elasticNet_mod3 <- caret::train(aux3$x_train, aux3$y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod3, highlight = TRUE)

elasticNet_mod3$bestTune$lambda # 0.04830804
elasticNet_mod3$bestTune$alpha # 0.1

kcvEnet_mod3 <- cv.glmnet(x=aux3$x_train, y=aux3$y_train, alpha=0.1, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod3 <- kcvEnet_mod3$glmnet.fit

# Predictions using the lambda.1se
elasticNet_res_mod3 <- summary_mod_performance(aux3$x_test, aux3$y_test, kcvEnet_mod3)



# Scenario 4 -------------------------------------------------------------------
simulations4 <- read.csv("syntheticData/scenario4.csv")
aux4 <- split_data(simulations4)

## Benchmark -------------------------------------------------------------------
ols4 <- lm(y~., data = aux4$train) # OLS
olsBIC_mod4 <- stepAIC(ols4, k=log(nrow(aux4$x_train)), trace=0) # OLS combined with the BIC


## Ridge (alpha=0) -------------------------------------------------------------
ridge_mod4 <- glmnet(aux4$x_train, aux4$y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod4 <- cv.glmnet(x=aux4$x_train, y=aux4$y_train, alpha=0, nfolds=10)
kcvRidge_mod4$lambda.min # 0.6629677
length(kcvRidge_mod4$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod4 <- 10^seq(log10(kcvRidge_mod4$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod4 <- cv.glmnet(x=aux4$x_train, y=aux4$y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod4)
kcvRidge2_mod4$lambda.min # 0.1
min(kcvRidge2_mod4$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod4$lambda.1se # 0.4119279

finalRidgeCV_mod4 <- kcvRidge2_mod4$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod4, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod4$lambda.min, kcvRidge2_mod4$lambda.1se)))

# Predictions using the lambda.1se
ridge_res_mod4 <- summary_mod_performance(aux4$x_test,aux4$y_test, kcvRidge2_mod4)


## Lasso (alpha=1) -------------------------------------------------------------
lasso_mod4 <- glmnet(x=aux4$x_train, y=aux4$y_train, alpha=1)
kcvLasso_mod4 <- cv.glmnet(x=aux4$x_train, y=aux4$y_train, alpha=1, nfolds=10)
kcvLasso_mod4$lambda.min
length(kcvLasso_mod4$lambda)
# 66, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod4 <- 10^seq(log10(kcvLasso_mod4$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod4 <- cv.glmnet(x=aux4$x_train, y=aux4$y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod4)
kcvLasso2_mod4$lambda.min # 0.1
min(kcvLasso2_mod4$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod4$lambda.1se # 0.2604009

finalLassoCV_mod4 <- kcvLasso2_mod4$glmnet.fit

# Predictions using the lambda.1se
lasso_res_mod4 <- summary_mod_performance(aux4$x_test, aux4$y_test, kcvLasso2_mod4)
predict(kcvLasso2_mod4, type = "coefficients",
        s = c(kcvLasso2_mod4$lambda.min, kcvLasso2_mod4$lambda.1se))
# There is difference!!


## Elastic Net ------------------------------------------------------------------
elasticNet_mod4 <- caret::train(aux4$x_train, aux4$y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod4, highlight = TRUE)

elasticNet_mod4$bestTune$lambda # 0.08723769
elasticNet_mod4$bestTune$alpha # 0.1

kcvEnet_mod4 <- cv.glmnet(x=aux4$x_train, y=aux4$y_train, alpha=0.1, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod4 <- kcvEnet_mod4$glmnet.fit

# Predictions using the lambda.1se
elasticNet_res_mod4 <- summary_mod_performance(aux4$x_test, aux4$y_test, kcvEnet_mod4)



# Scenario 5 -------------------------------------------------------------------
simulations5 <- read.csv("syntheticData/scenario5.csv")
aux5 <- split_data(simulations5)

## Benchmark -------------------------------------------------------------------
ols5 <- lm(y~., data=aux5$train) # OLS
olsBIC_mod5 <- stepAIC(ols5, k=log(nrow(aux5$x_train)), trace=0) # OLS combined with the BIC


## Ridge (alpha=0) -------------------------------------------------------------
ridge_mod5 <- glmnet(aux5$x_train, aux5$y_train, alpha=0)

# 10-fold cross-validation
kcvRidge_mod5 <- cv.glmnet(x=aux5$x_train, y=aux5$y_train, alpha=0, nfolds=10)
kcvRidge_mod5$lambda.min # 1.84889
length(kcvRidge_mod5$lambda) # 100 which is the last CV position
# Potential problem! To solve this issue, we extend the range

lambdaGrid1_mod5 <- 10^seq(log10(kcvRidge_mod5$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvRidge2_mod5 <- cv.glmnet(x=aux5$x_train, y=aux5$y_train, alpha=0, nfolds=10, lambda=lambdaGrid1_mod5)
kcvRidge2_mod5$lambda.min # 0.1
min(kcvRidge2_mod5$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvRidge2_mod5$lambda.1se # 0.2080346

finalRidgeCV_mod5 <- kcvRidge2_mod5$glmnet.fit
# Inspect the best models
plot(finalRidgeCV_mod5, label=TRUE, xvar="lambda")
abline(v = log(c(kcvRidge2_mod5$lambda.min, kcvRidge2_mod5$lambda.1se)))

# Predictions using the lambda.1se
ridge_res_mod5 <- summary_mod_performance(aux5$x_test, aux5$y_test, kcvRidge2_mod5)


## Lasso (alpha=1) -------------------------------------------------------------
lasso_mod5 <- glmnet(x=aux5$x_train, y=aux5$y_train, alpha=1)
kcvLasso_mod5 <- cv.glmnet(x=aux5$x_train, y=aux5$y_train, alpha=1, nfolds=10)
kcvLasso_mod5$lambda.min
length(kcvLasso_mod5$lambda)
# 69, lambda optimal is at the last position of the 10-CV
# Potential problem! To solve this issue, we extend the range

lambdaGrid2_mod5 <- 10^seq(log10(kcvLasso_mod5$lambda[1]), log10(0.1), length.out=150) # log-spaced grid

kcvLasso2_mod5 <- cv.glmnet(x=aux5$x_train, y=aux5$y_train, alpha=1, nfolds=10, lambda=lambdaGrid2_mod5)
kcvLasso2_mod5$lambda.min # 0.1
min(kcvLasso2_mod5$cvm) # The minimum CV error
# Search lambda 1SE for a trade-off
kcvLasso2_mod5$lambda.1se # 0.1576828

finalLassoCV_mod5 <- kcvLasso2_mod5$glmnet.fit

# Predictions using the lambda.1se
lasso_res_mod5 <- summary_mod_performance(aux5$x_test, aux5$y_test, kcvLasso2_mod5)
predict(kcvLasso2_mod5, type = "coefficients",
        s = c(kcvLasso2_mod5$lambda.min, kcvLasso2_mod5$lambda.1se))
# There is difference!!


## Elastic Net ------------------------------------------------------------------
elasticNet_mod5 <- caret::train(aux5$x_train, aux5$y_train, method="glmnet",
                                preProc=c("zv", "center", "scale"),
                                trControl=trainControl(method="cv", number=10),
                                tuneLength=10)
ggplot(elasticNet_mod5, highlight = TRUE)

elasticNet_mod5$bestTune$lambda # 0.01973394
elasticNet_mod5$bestTune$alpha # 0.2

kcvEnet_mod5 <- cv.glmnet(x=aux5$x_train, y=aux5$y_train, alpha=0.3, nfolds=10)
# REVIEW: should we fixed the lambda param also?
finalEnetCV_mod5 <- kcvEnet_mod5$glmnet.fit

# Predictions using the lambda.1se
elasticNet_res_mod5 <- summary_mod_performance(aux5$x_test, aux5$y_test, kcvEnet_mod5)



# Results ----------------------------------------------------------------------

mse_data <- data.frame(
  Scenario = rep(1:5, each=3),
  MSE = c(ridge_res_mod1$errors["MSEP"], lasso_res_mod1$errors["MSEP"], elasticNet_res_mod1$errors["MSEP"],
          ridge_res_mod2$errors["MSEP"], lasso_res_mod2$errors["MSEP"], elasticNet_res_mod2$errors["MSEP"],
          ridge_res_mod3$errors["MSEP"], lasso_res_mod3$errors["MSEP"], elasticNet_res_mod3$errors["MSEP"],
          ridge_res_mod4$errors["MSEP"], lasso_res_mod4$errors["MSEP"], elasticNet_res_mod4$errors["MSEP"],
          ridge_res_mod5$errors["MSEP"], lasso_res_mod5$errors["MSEP"], elasticNet_res_mod5$errors["MSEP"]),
  Method = rep(c("Ridge", "Lasso", "ElasticNet"), 15)
)

ggplot(mse_data, aes(x=factor(Scenario), y=MSE, fill=Method)) +
  geom_bar(stat="identity", position=position_dodge()) +
  labs(x="Scenario", y="Mean Squared Error") +
  scale_fill_brewer(palette = "Dark2") +
  theme_minimal()
