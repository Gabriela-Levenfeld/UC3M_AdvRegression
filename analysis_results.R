################################################################################
## Project: Advanced Regression and Prediction
##
## Script for Analysed results.
##
################################################################################

library(dplyr)
library(ggplot2)
library(pander)
library(tidyr)
library(forcats) #For plot coef of scenario 6
library(readr)

# Study MSE  ------------------------------------------------------------------
all_summary_stats <- list()
for (i in 1:6) {
  cat("Analysis of scenario ", i, "\n")
  
  # Load Scenario
  data_scenario <- read.csv(paste0("results/general_info/scenario", i, ".csv"), sep=",")
  
  # Create boxplot for MSEP
  p <- ggplot(data_scenario, aes(x=Method, y=MSEP, fill=Method)) +
    geom_boxplot() +
    labs(title=paste("Boxplot of MSEP - Scenario", i),
         x="Regularization Method", y="Mean Squared Error of Prediction") +
    scale_fill_brewer(palette="Dark2") +
    theme_minimal()
  
  p
  #ggsave(paste0("results/mse_boxplots/MSEP_boxplot_scenario", i, ".png"), plot=p) # Save the plot
  
  # Print statistics for Latex
  summary_stats <- data_scenario %>%
    group_by(Method) %>%
    summarise(
      Average_Error=mean(MSEP),
      Max_Error=max(MSEP),
      Min_Error=min(MSEP),
      Std_Deviation=sd(MSEP)
    )
  
  # Store them
  all_summary_stats[[i]] <- summary_stats
  pander(summary_stats)
}

# Print results for just one scenario of the six we have
scenario_to_see <- 1
all_summary_stats[[scenario_to_see]]

# Study coefficients ----------------------------------------------------------
results_list <- list()
for (i in 1:6) {
  cat("Analysis of scenario ", i, "\n")
  
  # Load Scenario
  data_scenario <- read.csv(paste0("results/general_info/scenario", i, ".csv"), sep=",")
  
  # Prepare data for analysis, melt it down for easier coefficient analysis
  coefficients_data <- data_scenario %>%
    dplyr::select(starts_with("X"), iteration, Method) %>%
    pivot_longer(cols = -c(iteration, Method), names_to = "Coefficient", values_to = "Value") %>%
    group_by(Method, Coefficient) %>%
    summarise(
      Zero_Count = sum(Value == 0, na.rm = TRUE),
      Near_Zero_Count = sum(abs(Value) < 0.005 & Value != 0, na.rm=TRUE),
      Average_Value = mean(Value, na.rm=TRUE),
      SD_Value = sd(Value, na.rm=TRUE),
      .groups = 'drop'
    )
  
  # Extract best lambda and alpha
  lambda_alpha_stats <- data_scenario %>%
    group_by(Method) %>%
    summarise(
      Average_Lambda = mean(best_lambda),
      Average_Alpha = mean(best_alpha),
      .groups = 'drop'
    )
  
  # Save results
  results_list[[i]] <- list(Coefficients = coefficients_data, LambdaAlphaStats = lambda_alpha_stats)
  #write.csv(results_list[[i]]$Coefficients, paste0("results/coef_summary/coef_summary_scenario", i, ".csv"))
  
  print(lambda_alpha_stats)
}

# Print results for just one scenario of the six we have
scenario_to_see <- 1
print(results_list[[scenario_to_see]]$Coefficients, n=33) # In scenario 6, n=120


# Scenario 6 is quite special
coef_data6 <- results_list[[6]]$Coefficients %>%
  mutate(Coefficient = fct_reorder(Coefficient, as.numeric(sub("X", "", Coefficient))))

ggplot(coef_data6, aes(x=Coefficient, y=Zero_Count, fill=Method)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.7)) +
  labs(x="Coefficient", y="Number of times coefficient is zero") +
  scale_fill_brewer(palette="Dark2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Plots coefficients -----------------------------------------------------------

## Scenario 1-3 ----
# Load real values
real_beta_1to3 <- read.csv("syntheticData/betas_s1tos4.csv", sep=",")
coefficients <- setdiff(paste0("X", 0:(length(real_beta_1to3$original_beta))), "X1")
real_beta_1to3 <- data.frame(
  Coefficient = coefficients,
  Average_Value = real_beta_1to3$original_beta,
  Method = "Actual",
  SD_Value = 0 # Just to avoid errors when combining
)

# Estimated coefficients
coef_data1 <- results_list[[3]]$Coefficients %>%
  mutate(Coefficient = fct_reorder(Coefficient, as.numeric(sub("X", "", Coefficient))))

# Combine for ggplot2: Select only the necessary columns to make structures identical
coef_data1 <- coef_data1 %>% select(Coefficient, Average_Value, Method, SD_Value)
real_beta_1to3 <- real_beta_1to3 %>% select(Coefficient, Average_Value, Method, SD_Value)
combined_data <- rbind(coef_data1, real_beta_1to3)

# ggplot(combined_data, aes(x=Coefficient, y=Average_Value, fill=Method)) +
#   geom_bar(stat="identity", position=position_dodge(width=0.7)) +
#   geom_errorbar(aes(ymin=Average_Value - SD_Value, ymax=Average_Value + SD_Value), width=.2, position=position_dodge(width=0.7)) +
#   labs(x="Coefficient", y="Average Value") +
#   scale_fill_brewer(palette="Dark2") +
#   theme_minimal()

ggplot(combined_data, aes(x=Coefficient, y=Average_Value, color=Method, group=Method)) +
  geom_line(size=0.8) +
  geom_point(size=2, shape=19) +
  labs(x="Coefficient", y="Average value") +
  scale_color_brewer(palette="Dark2") + # Use color for distinction
  theme_minimal() +
  theme(legend.position="bottom")



# ggplot(combined_data, aes(x=Coefficient, y=Average_Value, fill=Method)) +
#   geom_bar(stat="identity", position=position_dodge(width = 0.7)) +
#   labs(x="Coefficient", y="Average value") +
#   scale_fill_brewer(palette="Dark2") +
#   theme_minimal()

## Scenario 4 ----
real_beta_4 <- read.csv("syntheticData/betas_s1tos4.csv", sep=",")
coefficients <- setdiff(paste0("X", 0:(length(real_beta_4$beta_s4))), "X1")
real_beta_4 <- data.frame(
  Coefficient = coefficients,
  Average_Value = real_beta_4$beta_s4,
  Method = "Actual",
  SD_Value = 0 # Just to avoid errors when combining
)

# Estimated coefficients
coef_data4 <- results_list[[4]]$Coefficients %>%
  mutate(Coefficient = fct_reorder(Coefficient, as.numeric(sub("X", "", Coefficient))))

# Combine for ggplot2: Select only the necessary columns to make structures identical
coef_data4 <- coef_data4 %>% select(Coefficient, Average_Value, Method, SD_Value)
real_beta_4 <- real_beta_4 %>% select(Coefficient, Average_Value, Method, SD_Value)
combined_data <- rbind(coef_data4, real_beta_4)

ggplot(combined_data, aes(x=Coefficient, y=Average_Value, color=Method, group=Method)) +
  geom_line(size=0.8) +
  geom_point(size=2, shape=19) +
  labs(x="Coefficient", y="Average value") +
  scale_color_brewer(palette="Dark2") + # Use color for distinction
  theme_minimal() +
  theme(legend.position="bottom")

## Scenario 5 ----
real_beta_5 <- read.csv("syntheticData/betas_s5.csv", sep="")
coefficients <- setdiff(paste0("X", 0:(length(real_beta_5$x))), "X1")
real_beta_5 <- data.frame(
  Coefficient = coefficients,
  Average_Value = real_beta_5$x,
  Method = "Actual"
  )

coef_data5 <- results_list[[5]]$Coefficients %>%
  mutate(Coefficient = fct_reorder(Coefficient, as.numeric(sub("X", "", Coefficient))))

# Combine for ggplot2: Select only the necessary columns to make structures identical
coef_data5 <- coef_data5 %>% dplyr::select(Coefficient, Average_Value, Method)
combined_data5 <- rbind(coef_data5, real_beta_5)

ggplot(coef_data5, aes(x=Coefficient, y=Average_Value, color=Method, group=Method)) +
  geom_line(size=0.8) +
  geom_point(size=2, shape=19) +
  labs(x="Coefficient", y="Average value") +
  scale_color_brewer(palette="Dark2") + # Use color for distinction
  theme_minimal() +
  theme(legend.position="bottom")

## Scenario 6 ----
real_beta_6 <- read.csv("syntheticData/betas_s6.csv", sep="")
real_beta_6 <- data.frame(
  Coefficient = paste0("X", 0:40), # From X0 to X40 both included
  Average_Value = real_beta_6$beta_s6,
  Method = "Actual"
)

coef_data6 <- results_list[[6]]$Coefficients %>%
  mutate(Coefficient = fct_reorder(Coefficient, as.numeric(sub("X", "", Coefficient))))

# Combine for ggplot2: Select only the necessary columns to make structures identical
coef_data6 <- coef_data6 %>% select(Coefficient, Average_Value, Method)
combined_data6 <- rbind(coef_data6, real_beta_6)

ggplot(combined_data6, aes(x=Coefficient, y=Average_Value, color=Method, group=Method)) +
  geom_line(size=0.8) +
  geom_point(size=2, shape=19) +
  labs(x="Coefficient", y="Average value") +
  scale_color_brewer(palette="Dark2") + # Use color for distinction
  theme_minimal() +
  theme(legend.position="bottom")
