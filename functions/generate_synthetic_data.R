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


generate_data_correlated <- function(n, p, beta_coef, rho, noise_params) {
  # 1. Feature Matrix
  Sigma <- matrix(rho, nrow=p, ncol=p)
  diag(Sigma) <- 1 # Set the diagonal to 1 (correlation with itself)
  
  # Generate correlated feature matrix
  X_correlated <- mvrnorm(n, mu=rep(0, p), Sigma=Sigma)
  X_0 <- matrix(rep(1, n), ncol=1) # Add intercept
  X <- cbind(X_0, X_correlated)
  
  # 2. Creating Linear Relationship
  y_lin <- X %*% beta_coef
  
  # 3. Introducing Random Error (Noise) -> Follows a Normal distribution
  normal_error <- rnorm(n, mean=noise_params$mean, sd=noise_params$sd)
  y <- y_lin + normal_error
  
  # 4. Final Outputs
  data <- data.frame(y=y, X)
  data <- subset(data, select=-c(X1)) # Get rid off the intercept, no need anymore
  
  return(data)
}


generate_data_byGroups <- function(n, p, beta_coef, num_groups, group_means, noise_params) {
  obs_per_group <- n/num_groups
  group_data <- vector("list", num_groups)
  Sigma <- diag(p)
  
  # Generate data for each group
  for (i in 1:num_groups) {
    # 1. Feature Matrix: with group specific means
    X_group <- mvrnorm(obs_per_group, mu=group_means[i, ], Sigma=Sigma)
    X_0_group <- matrix(rep(1, obs_per_group), ncol=1) # Add the intercept
    X <- cbind(X_0_group, X_group)
    
    # 2. Creating Linear Relationship
    y_lin_group <- X %*% beta_coef
    
    # 3. Introducing Random Error (Noise) -> Follows a Normal distribution
    normal_error_group <- rnorm(obs_per_group, mean=noise_params$mean, sd=noise_params$sd)
    y_group <- y_lin_group + normal_error_group
    
    # Store the group data -> with the id group that the obs. belong
    group_data[[i]] <- data.frame(group=factor(i), y=y_group, X=X)
  }
  
  # 4. Final Outputs
  data <- do.call(rbind, group_data) # Combine in a df
  data <- subset(data, select = -c(X.1)) # Get rid off the intercept, no need anymore
  
  return(data)
}
