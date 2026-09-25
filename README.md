# OS580 Machine Learning

## Dataset Selection & Engineering Scope

### Module 0: Foundational Data Science Review
**File:** `Data Science Review.ipynb` | **Data Source:** `itunes_data.csv`

For this data prep assignment, I went with the Video Transcoding Dataset from the UCI Machine Learning Repository [I]. Working toward an AI video editing and workflow engineering role, cloud transcoding bottlenecks are a reality I think about a lot. When platforms ingest user media across random containers, you have to know how much compute power to spin up so export queues don't stall out. This dataset gives us 5,000 sampled rows, 20 features, and a target variable: `utime` (raw CPU transcoding time in seconds). It also gives us a solid mix of data types, pairing qualitative categories (`codec`, `o_codec`, `umem`) with continuous technical metrics like duration, bitrate, framerate, resolution dimensions, and frame types.

**Outlier Remediation via Interquartile Range (IQR) Clipping**
For outlier cleaning, I looked at the bitrate (bits per second). In real media pipelines, bitrate distributions get heavily skewed because high-bitrate master files produce massive throughput spikes compared to compressed web deliverables. My initial boxplot showed extreme upper outliers stretching all the way to 6.0 × 10⁶ bps. After running the IQR math, our upper boundary fence (Q3 + 1.5 × IQR) landed at roughly 1.43 × 10⁶ bps. Instead of simply deleting those rows, I used Pandas `.clip()` to cap values at that upper threshold. This keeps all 5,000 records intact, protects our statistical sample size, and prevents extreme spikes from throwing off downstream gradient descent or clustering steps.

**References**
[I] T. Deneke, H. A. Haile, P. Tsigas, and M. A. A. S. O. Z. R. S. H. K. S. M. A. F. M. H., "Video Transcoding Dataset," UCI Machine Learning Repository, 2017. doi: 10.24432/C5KP53.

---

### Week 1: Data Preparation & Unsupervised Clustering
**File:** `Assignment_1_Data_Prep_Clustering.ipynb` | **Data Sources:** `flightdelay.csv`, `US_births.csv`

**Part 1: Feature Engineering**
*   **High-Cardinality Filtering:** Identified and dropped `ID_TAG` based on a unique value ratio threshold greater than 0.6.
*   **Correlation Screening:** Evaluated linear associations against departure delay (`DEL15`), dropping statistically insignificant features (`AIRPORT_FLIGHTS_MONTH`, `PLANE_AGE` where p > 0.05). Computed non-linear Phi_K correlation and significance matrices.
*   **Distribution Remediation:** Addressed left-skewness and lower-tail outliers on `TMAX` via IQR lower-bound clipping.
*   **Scaling & Encoding:** Standardized `NUMBER_OF_SEATS` to zero mean and unit variance (`STAND_SEATS`) and mapped airline strings into discrete integers (`LE_CARRIER_NAME`) using `LabelEncoder`.

**Part 2: Unsupervised Clustering**
*   **K-Means Evaluation:** Tested k = 3 through 15 with Yellowbrick's `KElbowVisualizer` using silhouette scoring, profiling cluster density and knife-edge tapering with `SilhouetteVisualizer`.
*   **Dimensionality Reduction:** Fitted a k = 3 model and projected clusters into 2D space using Principal Component Analysis (PCA).
*   **Hierarchical Clustering:** Evaluated agglomerative distortion and runtime across k = 2 through 10, fitted final clusters with average linkage, and profiled median baselines for birth weight (`BWT`) and pre-pregnancy weight (`PWGT`).

---

### Week 2: Supervised Learning Pipelines & Model Evaluation
**File:** `MichaelDavid_First_Round_of_Supervised_Learning_Models.ipynb` | **Data Source:** `maintenance.csv`

*   **Data Splitting:** Executed a 70/30 stratified train-test split (random state `24`) on 7,518 rows to preserve the binary `Failure` class proportions.
*   **Logistic Regression:** Fitted an L1-penalized model (`liblinear` solver). Visualized exponentiated coefficients (odds ratios) in a sorted vertical bar plot, and parsed Statsmodels summaries to evaluate feature p-values.
*   **Naive Bayes:** Fitted a `GaussianNB` classifier and generated ROC-AUC curve visualizations with custom hex styling against random and perfect baseline models.
*   **K-Nearest Neighbors (KNN):** Configured a KNN classifier on the test set using *k*=10 and the `KDTree` algorithm. Evaluated performance via Scikit-Learn and Yellowbrick classification reports.
*   **Class Imbalance Mitigation:** Implemented random undersampling and bootstrapped oversampling to convert highly imbalanced target values into a clean 50/50 class distribution.

---

### Week 3: Advanced Regression, Optimization, & Feature Selection
**Discussion File:** `MichaelDavid_Week3_Initial_Discussion.ipynb` | **Data Source:** `flight_prices.csv`
**Assignment File:** `MichaelDavid_Assignment_Second_Round_of_Supervised_Learning_Models_3.ipynb` | **Data Source:** `cars.csv`

*   **Linear Regression Configs:** For the flight prices discussion, utilized `positive=True` to ensure logical, additive ticket pricing coefficients. For the car prices assignment, evaluated Statsmodels OLS (analyzing p-values to drop noisy features like `engine`) against a Scikit-Learn model forced through zero (`fit_intercept=False`), tracking resulting coefficient sign distortions.
*   **KNN Tuning:** Configured KNN Regressors using Manhattan distance (`p=1`) and `weights='distance'`. 
*   **Hyperparameter Optimization:** Built search dictionaries spanning neighbor counts, weights, and algorithms. Executed Grid Search (3-fold CV, neg RMSE) and Random Search (4-fold CV, neg MAE, 12 iterations), demonstrating how randomized subsets can uncover different optimal parameters than a rigid grid.
*   **Model Evaluation:** Manually calculated AIC and BIC scores to penalize unnecessary complexity. Plotted Yellowbrick Learning Curves on a new linear regression model to prove the training data had reached its performance ceiling.
*   **Feature Selection:** Pitted Recursive Feature Elimination (RFECV) against Backward Sequential Feature Selection (SFS). Output the raw boolean `.support_` arrays into a styled Pandas table to highlight algorithmic leniency differences.
*   **Industry Context:** Incorporated the 2026 MDS-VQA research citation to ground the statistical findings in real-world AI video engineering frameworks.

---
Week 4: Final Round of Supervised Learning Models & Peer Review
Discussion File: MichaelDavid_ResponceTo_OnyekachiOnyenokwe_Discussion4.ipynb | Data Source: flight_prices.csv
Assignment File: MichaelDavid_Assignment_Final_Round_of_Supervised_Learning_Models.ipynb | Data Sources: maintenance.csv, cars.csv

Decision Tree Classification: Implemented stratified train-test splits to predict binary machine failure. Configured dual DecisionTreeClassifier models (varying maximum depth, splitting criterion, and max features), utilizing Cohen's Kappa over raw accuracy to heavily penalize random chance guessing on the imbalanced dataset. Customized plot_tree outputs with dynamic F-string titles and localized node labeling.

Ensemble Optimization (Random Forest): Built a hyperparameter grid spanning estimator limits (50, 100, 200), purity criterions (Gini, Entropy), and depths. Executed a GridSearchCV evaluated via a weighted F1 scoring metric to optimize the Random Forest architecture against the maintenance dataset.

Decision Tree Regression: Configured a DecisionTreeRegressor (utilizing absolute error and Friedman MSE) on the cars dataset. Generated Yellowbrick FeatureImportances bar charts (enforcing custom brand colors) using topn=-5 to isolate and evaluate the bottom five least important parameters (identifying transmission as the least impactful node split).

Support Vector Regression (SVR) & Hyperplanes: As part of a collaborative peer review, standardized feature data using StandardScaler to ensure physical distance integrity before passing to an SVR algorithm. Tuned regularization (C), defined no-penalty "epsilon tube" boundaries, and styled the AIC/MSE outputs via a zebra-striped Pandas DataFrame.

SVM Kernel Variations: Benchmarked LinearSVR, a standard SVR (linear kernel), and a NuSVR (polynomial kernel degree 2) against the vehicle price dataset. Proved via MSE and calculated BIC tracking that polynomial hyperplanes vastly outperformed linear boundaries by successfully mapping the dataset's underlying non-linear correlations.

## Technical Stack
*   **Language:** Python 3.10+
*   **Environment:** Google Colab / JupyterLab
*   **Core Libraries:** Pandas, NumPy, Scikit-Learn, SciPy, Statsmodels, Matplotlib, Seaborn, Yellowbrick, Phi_K
*   **Version Control:** Git (Conventional Commits)

## Project Roadmap
[x] Week 1: Data Preparation & Unsupervised Clustering - Hypothesis Testing, Feature Engineering, K-Means & Hierarchical Clustering.

[x] Week 2: Supervised Learning Pipelines & Model Evaluation - Built Scikit-Learn/Statsmodels Logistic Regression, Gaussian Naive Bayes, KDTree KNN models, and class-imbalance resampling pipelines on the maintenance dataset.

[x] Week 3: Advanced Regression & Optimization - Configured OLS/Linear Regression and KNN models, performed Grid/Random Search tuning, plotted Yellowbrick learning curves, and executed RFECV/SFS feature selection on flight_prices and cars datasets.

[x] Week 4: Final Supervised Learning Models & SVM Implementation - Engineered Decision Tree Classification/Regression models, optimized Random Forest ensembles via Grid Search, customized Yellowbrick feature importance visualizations, and mapped non-linear correlations via polynomial kernel Support Vector Regressors.


Michael David
