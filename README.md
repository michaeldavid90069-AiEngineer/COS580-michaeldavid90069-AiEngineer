├── Data Science Review/
│   ├── Data Science Review.ipynb
│   └── itunes_data.csv
├── Week 1/
│   ├── Discussion - Data Prep/
│   │   ├── video_transcoding_sample.csv
│   │   ├── phik_target_correlation.png
│   │   ├── boxplot_before_clipping.png
│   │   └── boxplot_after_clipping.png
│   └── Assignment - Data Prep & Clustering/
│       ├── Assignment_1_Data_Prep_Clustering.ipynb
│       ├── flightdelay.csv
│       └── US_births.csv
└── README.md


Completed Implementations
Module 0: Foundational Data Science Review
File: Data Science Review.ipynb

Data Source: itunes_data.csv

Key Tasks:

Data ingestion, missing value imputation (Composer to Unknown), and deduplication.

Distribution assessment and outlier filtering on track lengths (Milliseconds) using IQR fence thresholds (Q1 - 1.5 * IQR to Q3 + 1.5 * IQR).

Feature transformation calculating playback time in Seconds and mean duration profiling grouped by Genre.

Visual inspection of cleaned durations using 30-bin Matplotlib histograms.

Week 1: Media Pipeline Optimization & Hypothesis Testing
File: Week 1 Discussion Submission

Data Source: UCI Video Transcoding Dataset (video_transcoding_sample.csv, 5,000 samples)

Key Tasks:

Continuous target identification on CPU transcoding time (utime in seconds) for cloud media workflow optimization.

Feature selection via non-linear Phi_K correlation and hypothesis testing (p-value threshold = 0.05).

Outlier mitigation on upper-tail bitrate distributions using IQR clipping (Q3 + 1.5 * IQR) with Pandas .clip() to retain total sample volume without downstream distortion.

Week 1: Data Preparation & Unsupervised Clustering
File: Assignment_1_Data_Prep_Clustering.ipynb

Part 1: Feature Engineering (flightdelay.csv)
High-Cardinality Filtering: Identified and dropped ID_TAG based on a unique value ratio threshold greater than 0.6.

Correlation Screening: Evaluated linear associations against departure delay (DEL15) with horizontal Pearson bar charts, dropping statistically insignificant features (AIRPORT_FLIGHTS_MONTH, PLANE_AGE where p > 0.05).

Non-Linear Dependencies: Computed Phi_K correlation and significance matrices; plotted dual bar charts utilizing custom turquoise green and pale violet red palettes with a bright yellow threshold line.

Distribution Remediation: Addressed left-skewness and lower-tail outliers on TMAX via IQR lower-bound clipping.

Scaling & Encoding: Standardized NUMBER_OF_SEATS to zero mean and unit variance (STAND_SEATS) and mapped airline strings into discrete integers (LE_CARRIER_NAME) using LabelEncoder.

Part 2: Unsupervised Clustering (US_births.csv)
Feature Scaling: Preprocessed continuous variables using StandardScaler.

K-Means Evaluation: Tested k = 3 through 15 with Yellowbrick's KElbowVisualizer using silhouette scoring, profiling cluster density and knife-edge tapering with SilhouetteVisualizer.

Dimensionality Reduction: Fitted a k = 3 model and projected clusters into 2D space using Principal Component Analysis (PCA), plotted with the Set1 colormap and black centroid markers.

Hierarchical Clustering: Evaluated agglomerative distortion and runtime across k = 2 through 10 using Yellowbrick, fitted final clusters with average linkage, and profiled median baselines for birth weight (BWT) and pre-pregnancy weight (PWGT).

Technical Stack
Language: Python 3.10+

Environment: Google Colab / JupyterLab

Core Libraries: Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn, Yellowbrick, Phi_K

Version Control: Git (Conventional Commits)

Project Roadmap
[x] Week 1: Data Preparation, Hypothesis Testing, Feature Engineering, K-Means & Hierarchical Clustering

[ ] Week 2: Supervised Learning Pipelines & Model Evaluation

[ ] Week 3: Advanced Classification, Hyperparameter Tuning & Cross-Validation

[ ] Week 4: Neural Architectures & End-to-End Pipeline Integration

Michael David
