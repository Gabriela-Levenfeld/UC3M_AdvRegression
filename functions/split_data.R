################################################################################
## Project: Advanced Regression and Prediction
##
## Function for splitting the data in train (80%) and test (20%) subsets
##
################################################################################

set.seed(1234) # for reproducibility

split_data <- function(data) {
  train_percentage <- 0.8 # 80% for training
  
  index <- floor(train_percentage * nrow(data))
  sample_train <- sample(1:nrow(data), index)
  sample_test <- (1:nrow(data))[-sample_train]
  
  # Split the data into training and testing sets
  train <- data[sample_train,]
  test <- data[sample_test,]
  
  # Names of the list are included, as train and test
  ouput <- list(train = train, test = test)
  
  return(ouput)
}