################################################################################
## Project: Advanced Regression and Prediction
##
## Function for splitting the data in train (80%) and test (20%) subsets, and
## prepare data output for modelling.
##
################################################################################

#set.seed(1234) # for reproducibility

split_data <- function(data) {
  train_percentage <- 0.8 # 80% for training
  
  index <- floor(train_percentage * nrow(data))
  sample_train <- sample(1:nrow(data), index)
  sample_test <- (1:nrow(data))[-sample_train]
  
  # Split the data into training and testing sets
  train <- data[sample_train,]
  test <- data[sample_test,]
  
  # Prepare data
  x_train <- as.matrix(subset(train, select=-c(y)))
  x_test <- as.matrix(subset(test, select=-c(y)))
  
  y_train <- train$y
  y_test <- test$y
  
  ouput <- list(train=train, x_train=x_train, y_train=y_train,
                test=test, x_test=x_test, y_test=y_test)
  
  return(ouput)
}
