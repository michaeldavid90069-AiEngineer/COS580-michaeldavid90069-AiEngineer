# Week 3 Discussion: Initial Regression Models (Flight Prices)

## Overview
This archive contains the initial Jupyter Notebook and dataset for the Week 3 Discussion board. The objective of this project is to build and configure two regression models—Linear Regression and K-Nearest Neighbors (KNN)—to predict continuous flight prices based on encoded flight features. 

**Important Note for Peer Reviewers:** As strictly outlined in the assignment instructions and the Week 3 lecture, this notebook halts execution immediately after the models are fitted (`.fit()`). No scoring metrics (such as R-squared, MSE, or MAE) have been calculated. These models are primed and ready for you to generate the performance metrics and declare a winner in your response post!

## Contents
* `MichaelDavid_Week3_Initial_Discussion.ipynb`: The executed Jupyter Notebook containing the data splitting and model fitting pipeline.
* `flight_prices.csv`: The pre-encoded dataset containing 29,813 flight records and 10 columns (9 features, 1 continuous target `price`).

## Methodology & Hyperparameter Selections
1. **Data Splitting:** 
   * The dataset was split into 80% training data and 20% test data. 
   * A fixed random state (`random_state=42`) was used for reproducibility. 
   * Stratification was intentionally excluded, as the target variable is continuous.
2. **Linear Regression:** 
   * Built using `sklearn.linear_model.LinearRegression`.
   * **Hyperparameter tweak:** `positive=True` (default is False). This forces the model to assign only positive coefficients, aligning with the logic that additional flight factors (like duration) should additively increase ticket prices.
3. **K-Nearest Neighbors (KNN) Regression:**
   * Built using `sklearn.neighbors.KNeighborsRegressor`.
   * **Hyperparameter tweaks:** `n_neighbors=7` (default is 5), `weights='distance'` (closer neighbors have a higher mathematical influence), and `p=1` (utilizing Manhattan distance instead of the default Euclidean distance).

## Author
Michael David