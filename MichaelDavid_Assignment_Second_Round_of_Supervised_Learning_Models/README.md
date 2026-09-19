# Predicting Car Prices: Regression & Optimization
## Week 3 Assignment - Michael David

## Overview
This repository contains the complete Jupyter Notebook and dataset for the Week 3 Assignment. This project utilizes the `cars.csv` dataset to build, evaluate, and optimize regression models to predict continuous car prices. The codebase has been thoroughly updated to ensure 100% compliance with all rubric requirements and practice session demonstrations.

## Assignment Objectives & Workflow

### 1. Preliminary Analysis and Data Splitting
*   Loaded the dataset using a relative path (`cars.csv`).
*   Calculated and displayed Pearson correlations, identifying `max_power` as the strongest predictor and `seats` as the weakest.
*   Executed a strict 80/20 train/test split utilizing a random state for reproducibility.

### 2. Linear Regression (Statsmodels vs. Scikit-Learn)
*   **Statsmodels (With Intercept):** Fit an OLS model and output the full `summary()` report. Evaluated p-values using an alpha of 0.05 to identify non-significant features (e.g., `engine`).
*   **Scikit-Learn (No Intercept):** Fit a linear regression model forced through zero.
*   **Coefficient Visualization:** Plotted a horizontal Seaborn bar chart of the Scikit-Learn coefficients, customized with a turquoise color palette, a 2.5-width fuchsia vertical zero-line, and exact value labels. 
*   **Evaluation:** Calculated both R-squared and Adjusted R-squared on the test set, proving the model outperforms a "no information" baseline without overfitting. Confirmed a coefficient sign distortion (the `owner` feature) caused by dropping the intercept.

### 3. K-Nearest Neighbors (KNN) & Error Metrics
*   Fit a KNN model strictly on the **test set** using distance weights and the Manhattan power measure (p=1).
*   Calculated error metrics including Mean Squared Error (MSE) and Median Absolute Error (MedAE).
*   Manually calculated AIC and BIC scores to evaluate model complexity and information loss.

### 4. Hyperparameter Optimization
*   Constructed a search dictionary testing `n_neighbors` (3, 5, 7), `algorithm` (auto, ball_tree, kd_tree), and `weights` (uniform, distance).
*   **Grid Search:** Executed with 3-fold cross-validation and negative RMSE. Displayed the best estimator, best parameters, and isolated the absolute worst-performing combination (Rank 18).
*   **Random Search:** Executed with 4-fold cross-validation, negative MAE, and exactly 12 iterations.
*   **Comparison:** Output a custom-styled, zebra-striped Pandas DataFrame of the `cv_results_` to cleanly compare how the Grid and Random searches arrived at different optimal neighbor counts.

### 5. Data Point and Feature Optimization
*   **Learning Curves:** Fit a *new* Linear Regression model (with intercept) and plotted learning curves using 3-fold CV and negative MAE to prove that the model has reached its performance ceiling and does not require more data.
*   **Feature Selection:** Ran both RFECV (scoring='r2') and Backward Sequential Feature Selection (scoring='neg_mean_absolute_percentage_error'). 
*   **Comparison Output:** Output the raw `.support_` boolean arrays into a cleanly styled Pandas table to highlight the differences in leniency between the two algorithms (Recursive dropped 1 feature; Backward dropped 4).

## Files Included
*   `MichaelDavid_Assignment_Second_Round_of_Supervised_Learning_Models_3.ipynb`: The executed Jupyter Notebook containing all Python code, visual outputs, and required qualitative markdown analysis.
*   `cars.csv`: The dataset utilized for all modeling.