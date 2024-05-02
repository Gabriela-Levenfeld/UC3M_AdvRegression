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


# Load required libraries ------------------------------------------------------
library(glmnet)
library(caret)

# Set up all parameters used ---------------------------------------------------
set.seed(1234) # for reproducibility
n <- 100 # number of observations
p <- 10 # number of predictors


# Synthetic data generation ----------------------------------------------------
# 1. Generate random values for the betas (as uniform integers in [-10,10])
beta_coef <- runif(p+1, min = -10, max = 10) # Include beta_0 for the intercept

print('Values of the beta coefficients:')
for (i in 1:(p+1)){
  print(beta_coef[i])
}

## Scenario 1 ------------------------------------------------------------------
# 2. Generating Feature Matrix

# Each element is drawn independently:
# No correlation between variables, X
# Follows a Normal distribution (0, 1)
mu_X <- 0; sigma_X <- 0.5
X_0 <- matrix(rep(1, n), ncol = 1)
X_1 <- matrix(rnorm(n*p, mean = mu_X, sd = sigma_X), ncol = p)
X <- cbind(X_0, X_1)

# To introduce correlation between predictors
#X <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma)

# 3. Creating Linear Relationship
y_lin <- X %*% beta_coef

# 4. Introducing Random Error (Noise)
rand_error <- stats::rnorm(n, mean = 0, sd = 0.1)
y <- y_lin + rand_error

# 5. Final Outputs
data1 <- data.frame(y = y, X)


## Preparing the dataset --------------------------------------------------------

mcor <- cor(data1)
mcor
abs(mcor)>0.8
corrplot::corrplot(mcor, method = "ellipse")

index <- floor(0.8*nrow(data1))
sample_train <- sample(1:nrow(data1), index)
sample_test <- (1:nrow(data1))[-sample_train]

train <- data1[sample_train,]
test <- data1[sample_test,]

y_train <- train$y
y_test <- test$y

x_train <- subset(train, select = -c(y))
x_train <- as.matrix(x_train)
x_test <- subset(test, select = -c(y))
x_test <- as.matrix(x_test)

## OLS ---------------------
ols <- lm(y ~ ., data = train)

## Ridge regression ------------------
ridge_mod1 <- glmnet(x_train, y_train, alpha =0)#[,-1]#We exclude the intercept.
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

predRidge<- predict(ridge_mod1, newx = newx)#s=lambda.1SE
predLasso<- predict(kcvLasso, newx = newx)
predEnet<-predict(elasticnet,newx=newx)
predols<- predict(modOLSBIC, newdata =test)
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
