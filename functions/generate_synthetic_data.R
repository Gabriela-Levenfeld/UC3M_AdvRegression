################################################################################
## Project: Advanced Regression and Prediction
##
## Some functions to generate the synthetic data with different specifications.
##
################################################################################

library(MASS) # X matrix for creating predictors correlated

generate_data_indep <- function(n, p, beta_coef, X_matrix, X_params, noise_type, noise_params) {
  # 1. Feature Matrix
  if (is.null(X_matrix)) {
    # Generate Feature Matrix from scratch
    X_0 <- matrix(rep(1, n), ncol=1) # Add intercept
    X_1 <- matrix(rnorm(n*p, mean=X_params$X_mu, sd=X_params$X_sigma), ncol=p)
    X <- cbind(X_0, X_1)
  } else {
    # Download the already provided X_matrix (is passed as df)
    X_0 <- matrix(rep(1, n), ncol = 1, dimnames = list(NULL, "X1")) # Add intercept with name X1
    X_1 <- as.matrix(X_matrix) # Force to be a matrix
    X <- cbind(X_0, X_1) # We need to include the intercept column
  }

  # 2. Creating Linear Relationship
  y_lin <- X %*% beta_coef
  
  # 3. Introducing Random Error (Noise)
  if (noise_type == "normal") {
    rand_noise <- rnorm(n, mean=noise_params$mean, sd=noise_params$sd)
  } else if (noise_type == "exponential") {
    rand_noise <- rexp(n, rate = noise_params$rate)
  }
  y <- y_lin + rand_noise
  
  # 4. Final Outputs
  data <- data.frame(y=y, X)
  data <- subset(data, select=-c(X1)) # Get rid off the intercept, no need anymore
  
  return(data)
}


generate_correlated_data_groups <- function(n, p, group_size = 3) {
  X <- matrix(rnorm(n*p), ncol=p)
  
  # Adding correlation to the first group by a common noise
  noise_first_group <- rnorm(n) 
  for (j in 1:group_size) {
    X[, j] <- X[, j] + noise_first_group
  }
  
  # Adding correlation to the first block by a group noise, if it is possible
  if (2 * group_size <= p) {
    noise_second_group=rnorm(n)
    for (j in 1:group_size) {
      X[, group_size + j] <- X[, group_size + j] + noise_second_group
    }
  }
  
  # Generate normal random coefficients for beta
  beta <- rnorm(p)
  
  # The noise follows a normal distribution N(0,1)
  y <- X %*% beta + rnorm(n)
  
  data <- data.frame(y=y, X)
  
  output <- list(data=data, beta_coef=beta)
  return(output)
}
