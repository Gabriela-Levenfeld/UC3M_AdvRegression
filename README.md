# Comparison of regularization methods

**Subject**: Advanced Regression and Prediction at Universidad Carlos III de Madrid (UC3M).

**Abstract**

This study explores the performance of three regularization techniques - ridge regression, lasso, and elastic net - through a simulation study. Six scenarios  were created under various circumstances such as multicollinearity, different noise distributions, and high dimensional data, in order to carry out a comparative analysis. To do so, we assess each method's capacity for variable selection as well as the error (in terms of MSEP) made on unseen data, highlighting the limitations of each penalization approach.

Project jointly developed with [Miguel Díaz-Plaza Cabrera](https://github.com/migueldiazpl).

**Programming language**: 
  ![R Version](https://img.shields.io/badge/R-4.3.1-blue.svg) 

## Repository Structure

```plaintext
/
├── functions/ # Contains the functions implemented for doing the analyses
├── results/ # Contains summary CSV files and visualization outputs
│ ├── general_info/ # All information collected from each scenario
│ ├── coef_summary/ # Summary files of the variable selection
│ ├── mse_boxplots/ # Boxplots for the MSE on test sets
├── syntheticData/ # Directory containing all generated datasets
├── simulation_scenarios.R # Script for setting up the simulation scenarios
├── regularization_procedures.R # Script for apply the three regularization techniques
├── analysis_results.R # Script for analyzing results from the regularization techniques
```
