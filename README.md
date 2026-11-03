# Obesity Data Analytics

A data analytics and machine learning project exploring obesity-related characteristics through data preprocessing, exploratory data analysis, unsupervised clustering, classification, and BMI regression.

## 1. Project Overview

This project investigates relationships between demographic characteristics, eating habits, physical activity, and obesity categories using a structured dataset containing 2,111 observations.

The main objective is to apply data analytics and machine learning techniques to identify patterns in lifestyle-related characteristics and evaluate the predictive performance of different classification and regression models.

The project consists of four main analytical stages:

- **Data Preprocessing and Exploratory Data Analysis:** Data cleaning, numerical recovery, statistical analysis, feature transformation, and visualization.
- **Clustering:** Application of K-Means and DBSCAN to investigate potential lifestyle patterns.
- **Classification:** Prediction of obesity categories using Random Forest and Gaussian Naive Bayes classifiers.
- **Regression:** Prediction of Body Mass Index (BMI) using Random Forest and Feedforward Neural Networks.

The project was developed as part of a university assignment and subsequently refined to improve its preprocessing methodology, evaluation procedures, and documentation.

## 2. Dataset Description

The dataset is provided in an Excel file (`Data1.xlsx`), using the worksheet `Φύλλο2`.

It contains:

- **2,111 observations**
- **17 columns**
- **8 numerical features**
- **9 categorical features**, including the target variable.

### Features

| Feature | Description |
|---|---|
| Gender | Gender of the individual |
| Age | Age in years |
| Height | Height in metres in the original dataset; recovered and converted to centimetres during preprocessing |
| Weight | Body weight in kilograms |
| family_history_with_overweight | Whether the individual has a family history of overweight |
| FAVC | Frequent consumption of high-caloric food |
| FCVC | Frequency of vegetable consumption |
| NCP | Number of main meals per day |
| CAEC | Consumption of food between meals |
| SMOKE | Smoking status |
| CH2O | Daily water consumption |
| SCC | Monitoring of calorie consumption |
| FAF | Frequency of physical activity |
| TUE | Time spent using technological devices |
| CALC | Alcohol consumption frequency |
| MTRANS | Main mode of transportation |
| NObeyesdad | Obesity category (target variable) |

### Target Variable

`NObeyesdad` contains seven categories:

1. Insufficient_Weight
2. Normal_Weight
3. Obesity_Type_I
4. Obesity_Type_II
5. Obesity_Type_III
6. Overweight_Level_I
7. Overweight_Level_II

## 3. Data Preprocessing and Exploratory Data Analysis

### 3.1 Data Cleaning and Numerical Recovery

The initial preprocessing stage addressed inconsistencies in the numerical values and formats of the dataset.

The following operations were performed:

- Conversion of numerical values stored in inconsistent formats, including decimal separators represented by commas.
- Identification and recovery of Height values affected by inconsistent numerical scales.
- Conversion of recovered height measurements into centimetres.
- Validation of numerical features for missing, invalid, and suspicious values.
- Examination of Height and Weight ranges to identify implausible measurements.

Because Height values exhibited different numerical scales, several heuristic recovery methods were applied. These included division by powers of ten and conversion of shorter values into centimetres.

The final dataset contained **2,111 records**, with no records removed during the latest execution.

All selected numerical features contained valid values after preprocessing.

**Data recovery limitation:** Height recovery is based on numerical scaling heuristics and plausibility checks. Although the resulting measurements fall within plausible ranges, the procedure cannot guarantee exact reconstruction of every original measurement.

### 3.2 Statistical Analysis

Descriptive statistics were calculated to examine the distributions of numerical variables.

Selected statistics after preprocessing:

| Feature | Mean | Standard Deviation |
|---|---:|---:|
| Age | 24.31 | 6.35 |
| Height (cm) | 170.17 | 9.33 |
| Weight (kg) | 86.59 | 26.19 |
| FCVC | 2.42 | 0.53 |
| NCP | 2.69 | 0.78 |
| CH2O | 2.01 | 0.61 |
| FAF | 1.16 | 0.84 |
| TUE | 0.74 | 0.61 |

BMI was calculated using the recovered height and weight measurements:

`BMI = Weight / (Height / 100)²`

The resulting BMI distribution had:

- Mean: **29.70**
- Standard deviation: **8.01**
- Minimum: **13.00**
- Maximum: **50.81**

BMI statistics were also examined across the seven obesity categories to assess their numerical consistency.

### 3.3 Data Transformation

For the general exploratory and clustering workflow:

- Numerical features were standardized using `StandardScaler`.
- Categorical variables were encoded using `LabelEncoder`.

For the classification and regression experiments, separate preprocessing pipelines were implemented using `ColumnTransformer`.

These pipelines applied:

- `StandardScaler` to numerical predictors.
- `OneHotEncoder` to categorical predictors.

Importantly, preprocessing for the supervised models was fitted exclusively on the training data and subsequently applied to the test data, preventing information leakage during preprocessing.

### 3.4 Exploratory Data Visualizations

The exploratory analysis included:

- Histograms to examine numerical feature distributions.
- Boxplots to investigate potentially unusual observations.
- Correlation heatmaps to examine relationships between numerical variables.
- Principal Component Analysis (PCA) to explore the structure of the numerical feature space and support dimensionality reduction.

The first two PCA components explained approximately **46.45% of the total variance** in the selected standardized lifestyle features.

## 4. Clustering Analysis

### 4.1 Objective and Feature Selection

Unsupervised clustering was applied to investigate potential patterns in dietary habits, water consumption, and physical activity.

Five lifestyle-related features were selected:

| Feature | Description |
|---|---|
| FCVC | Frequency of vegetable consumption |
| NCP | Number of main meals per day |
| CH2O | Daily water consumption |
| FAF | Frequency of physical activity |
| TUE | Time spent using technological devices |

The selected features were standardized before clustering.

### 4.2 K-Means Clustering

K-Means was evaluated for values of K ranging from 2 to 10.

The Silhouette Score was calculated for each configuration to assess cluster separation.

| Number of clusters (K) | Silhouette Score |
|---:|---:|
| 2 | **0.239** |
| 3 | 0.206 |
| 4 | 0.193 |
| 5 | 0.199 |
| 6 | 0.203 |
| 7 | 0.200 |
| 8 | 0.213 |
| 9 | 0.208 |
| 10 | 0.216 |

The best result was obtained with **K = 2**, achieving a Silhouette Score of **0.239**.

The resulting cluster distribution was:

| Cluster | Observations | Percentage |
|---|---:|---:|
| Cluster 0 | 433 | 20.5% |
| Cluster 1 | 1,678 | 79.5% |

### 4.3 K-Means Cluster Interpretation

Cluster profiles were examined using the original, unstandardized feature values.

| Feature | Cluster 0 | Cluster 1 |
|---|---:|---:|
| FCVC | 2.358 | 2.435 |
| NCP | 1.294 | 3.045 |
| CH2O | 1.927 | 2.029 |
| FAF | 1.095 | 1.181 |
| TUE | 0.652 | 0.767 |

The most noticeable difference between the two groups concerns `NCP`, representing the number of main meals per day.

Cluster 0 reported approximately 1.29 main meals per day, whereas Cluster 1 reported approximately 3.05.

The remaining lifestyle characteristics were relatively similar.

Although K-Means identified two groups, the Silhouette Score of 0.239 indicates relatively weak cluster separation. Therefore, these groups should be interpreted as broad patterns in the selected features rather than clearly distinct behavioural profiles.

### 4.4 DBSCAN Clustering

DBSCAN was also implemented to investigate density-based grouping and identify observations that could not be assigned to sufficiently dense regions.

The selected parameters were:

- `eps = 0.6`
- `min_samples = 4`

The resulting Silhouette Score was **-0.166**, calculated on the non-noise observations.

DBSCAN identified:

- 1,228 observations in its largest cluster.
- 478 observations classified as noise (`label = -1`).
- Numerous additional small clusters.

The negative Silhouette Score and fragmented cluster distribution indicate that this configuration did not produce a particularly useful clustering structure.

### 4.5 Clustering Visualization and Evaluation

The clustering analysis included:

- Silhouette Score comparisons across different K-Means configurations.
- Cluster distribution analysis.
- PCA-based two-dimensional projections.
- Scatter plots illustrating the resulting clusters.
- Execution-time measurements.
- Comparison of the clustering assignments with the actual obesity categories (`NObeyesdad`).

The comparison with the actual categories provides additional context for interpreting the unsupervised results, but does not change the unsupervised nature of the clustering algorithms.

### 4.6 Clustering Conclusions

K-Means produced a more interpretable result than the selected DBSCAN configuration.

However, both approaches demonstrated limitations in identifying strongly separated lifestyle groups.

The results suggest that the selected lifestyle features alone do not form clearly separated clusters in this dataset.

## 5. Classification: Obesity Category Prediction

### 5.1 Objective

The classification task aims to predict the obesity category (`NObeyesdad`) using demographic, physical, and lifestyle characteristics.

Three classification experiments were conducted:

1. Random Forest using all available predictors, including Height and Weight.
2. Random Forest excluding Height and Weight.
3. Gaussian Naive Bayes using all available predictors.

The second Random Forest experiment was included to examine how predictive performance changes when direct physical measurements are excluded.

### 5.2 Data Preparation

The target variable was separated from the predictor variables.

The dataset was divided into:

- **80% training data**
- **20% testing data**

A stratified train/test split was used to preserve the distribution of the seven target classes.

Preprocessing was implemented through Scikit-learn pipelines:

- Numerical features were standardized using `StandardScaler`.
- Categorical features were transformed using `OneHotEncoder`.
- Preprocessing transformations were fitted only on the training data.

The same stratified split configuration was used for the two Random Forest experiments to support a fair comparison.

### 5.3 Classification Models

**Random Forest Classifier**

Random Forest was used to model potentially complex and nonlinear relationships between the input characteristics and obesity categories.

Two versions were evaluated:

- **Model A:** Includes Height and Weight.
- **Model B:** Excludes Height and Weight.

**Gaussian Naive Bayes**

Gaussian Naive Bayes was implemented as a probabilistic classification baseline using the same general feature set as Random Forest Model A.

### 5.4 Evaluation Metrics

The classifiers were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Multiclass ROC curves using a One-vs-Rest approach

Macro-averaged F1-score was included to provide an evaluation that gives equal importance to each target class.

### 5.5 Classification Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Random Forest (with Height and Weight) | **0.93** | 0.94 | 0.93 | **0.93** |
| Random Forest (without Height and Weight) | 0.85 | 0.85 | 0.85 | 0.85 |
| Gaussian Naive Bayes | 0.51 | 0.52 | 0.50 | 0.45 |

### 5.6 Classification Analysis

Random Forest achieved the best classification performance.

The model using Height and Weight achieved 93% accuracy and a macro F1-score of 0.93.

When Height and Weight were excluded, accuracy decreased to 85%, with a macro F1-score of 0.85.

This difference indicates that these physical measurements contain substantial predictive information about the obesity categories.

This is an expected result because obesity categories are closely related to body-weight characteristics and BMI.

The Random Forest model without Height and Weight nevertheless maintained relatively strong predictive performance, suggesting that demographic and lifestyle-related characteristics also contain useful information.

Gaussian Naive Bayes achieved lower performance, with 51% accuracy and a macro F1-score of 0.45.

The confusion matrices and class-specific evaluation metrics provide additional insight into differences in predictive performance across the seven categories.

## 6. Regression: BMI Prediction

### 6.1 Objective

The regression task aims to predict Body Mass Index (BMI) as a continuous numerical variable.

BMI was calculated from the Height and Weight measurements:

`BMI = Weight / (Height / 100)²`

Unlike the classification task, the regression target is a numerical value rather than an obesity category.

Height and Weight were excluded from the regression predictors because they are directly used to calculate the target variable.

### 6.2 Feature Selection and Preprocessing

The regression models used demographic and lifestyle-related predictors.

Numerical features:

- Age
- FCVC
- NCP
- CH2O
- FAF
- TUE

Categorical features:

- Gender
- family_history_with_overweight
- FAVC
- CAEC
- SMOKE
- SCC
- CALC
- MTRANS

The dataset was divided into 80% training data and 20% testing data.

Preprocessing was performed using a `ColumnTransformer`:

- `StandardScaler` for numerical features.
- `OneHotEncoder` for categorical features.

The transformations were fitted on the training data and applied to the test data.

The resulting feature matrices contained 29 transformed features:

- Training set: 1,688 observations.
- Testing set: 423 observations.

Data diagnostics confirmed that the transformed training and testing matrices contained no missing or infinite values.

### 6.3 Regression Models

Four regression approaches were evaluated.

**Dummy Regressor**

A baseline model using the training target's mean as its prediction.

It provides a reference point for determining whether the machine learning models offer meaningful improvements over a simple prediction strategy.

**Random Forest Regressor**

An ensemble-based regression model using 200 decision trees and `random_state=42`.

**Feedforward Neural Network 1**

The first neural network was implemented using TensorFlow/Keras.

Architecture:

- Input layer corresponding to the transformed features.
- First hidden layer: 64 neurons with ReLU activation.
- Second hidden layer: 32 neurons with ReLU activation.
- Output layer: one neuron with linear activation.

**Feedforward Neural Network 2**

A second independently trained neural network using the same general architecture.

It was evaluated to compare the performance of another training run, rather than to implement transfer learning.

### 6.4 Regression Evaluation Metrics

The regression models were evaluated using:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted BMI values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors more heavily.
- **MAPE (Mean Absolute Percentage Error):** Measures average absolute percentage error.
- **R² (Coefficient of Determination):** Measures the proportion of target variance explained by the model.

Training and validation loss curves were also generated for the neural network experiments.

### 6.5 Regression Results

| Model | MAE | RMSE | MAPE | R² |
|---|---:|---:|---:|---:|
| Dummy Regressor | 6.849 | 8.128 | — | ≈ 0.000 |
| Random Forest Regressor | **1.914** | **2.890** | — | **0.874** |
| Feedforward Neural Network 1 | 2.961 | 3.727 | 11% | 0.790 |
| Feedforward Neural Network 2 | 2.779 | 3.604 | 10% | 0.803 |

### 6.6 Regression Analysis

Random Forest achieved the best overall regression performance.

With an MAE of 1.914 and an RMSE of 2.890, it substantially outperformed the Dummy Regressor baseline.

Its R² score of 0.874 indicates that it explained approximately 87.4% of the variation in BMI within the test set.

Both Feedforward Neural Networks also outperformed the baseline.

The second neural network achieved slightly better results than the first, with an MAE of 2.779 and an R² of 0.803.

However, neither neural network outperformed Random Forest in this experiment.

These findings demonstrate that the Random Forest Regressor was the most effective of the evaluated models for predicting BMI from the selected demographic and lifestyle characteristics.

## 7. Overall Findings

The project explored both unsupervised and supervised machine learning approaches.

The main findings were:

- Data preprocessing and heuristic numerical recovery produced a complete dataset containing 2,111 observations.
- PCA provided a two-dimensional representation explaining 46.45% of the variance in the selected lifestyle features.
- K-Means with two clusters produced the best Silhouette Score among the tested configurations, although the separation remained weak.
- The selected DBSCAN configuration produced fragmented clusters and a negative Silhouette Score.
- Random Forest achieved the best classification performance, reaching 93% accuracy when Height and Weight were included.
- Excluding Height and Weight reduced Random Forest accuracy to 85%, demonstrating the predictive contribution of these measurements.
- Random Forest also achieved the best BMI regression performance, with an MAE of 1.914 and an R² of 0.874.
- Both Feedforward Neural Networks outperformed the regression baseline but did not surpass Random Forest.

Overall, the experiments demonstrate the application of data preprocessing, exploratory analysis, clustering, classification, and regression to a structured obesity-related dataset.

## 8. Technologies and Libraries

The project was implemented in Python using the following technologies:

| Technology / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading, manipulation, and analysis |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, clustering, classification, regression, and evaluation |
| TensorFlow / Keras | Feedforward neural network implementation |
| OpenPyXL | Reading Excel files |

## 9. Project Structure

```text
Obesity Data Analytics/
│
├── Data/
│   └── Data1.xlsx
│
├── plots/
│   └── Generated visualizations
│
├── data_analytics.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

The `.venv` directory is used for local Python environment management and should not be included in the Git repository.

## 10. Installation and Execution

### Prerequisites

- Python installed on your system.
- Git (optional, for repository management).

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Prepare the dataset

Place the Excel dataset in the following location:

```text
Data/Data1.xlsx
```

Ensure that the worksheet name is `Φύλλο2`, as expected by the script.

The dataset is required for execution and must be available at the expected path.

### 6. Run the project

```bash
python data_analytics.py
```

The script performs the preprocessing, exploratory analysis, clustering, classification, and regression workflows.

Generated visualizations and evaluation plots are saved in the `plots/` directory.

## 11. Limitations and Considerations

Several limitations should be considered when interpreting the results:

- **Heuristic data recovery:** The Height recovery procedure relies on numerical scaling and plausibility checks. It cannot establish the exact original measurement for every observation.
- **Clustering quality:** The relatively low K-Means Silhouette Score and negative DBSCAN Silhouette Score indicate limited cluster separation. The resulting groups should not be interpreted as definitive behavioural categories.
- **Feature-target relationship:** Height and Weight contribute directly to obesity-category prediction. Consequently, the classification results using these features should not be interpreted as independent prediction based solely on lifestyle information.
- **BMI target construction:** BMI is mathematically derived from Height and Weight. Although these measurements were excluded from the regression predictors, the target remains dependent on their accuracy.
- **Dataset scope:** The analysis is based on the available dataset and does not establish causal relationships between lifestyle characteristics and obesity.
- **Model generalization:** Evaluation was performed using a single stratified 80/20 train/test split. Further validation using cross-validation and independent datasets would provide additional evidence of generalization.

The reported results describe the evaluated dataset and experimental configuration; they should not be interpreted as evidence of clinical or real-world diagnostic performance.

## 12. Future Improvements

Potential extensions of the project include:

- Further investigation of numerical data recovery and validation.
- Testing alternative clustering algorithms and parameter configurations.
- Applying cross-validation and more extensive hyperparameter tuning.
- Investigating feature importance and model interpretability.
- Exploring additional regression algorithms and neural network architectures.
- Evaluating model generalization using independent datasets.

## 13. License and Dataset Attribution

The dataset is provided separately from the analysis code.

Before redistributing `Data1.xlsx` through a public GitHub repository, its original source, attribution requirements, and redistribution permissions should be verified.

The project code and the dataset may be subject to different licensing conditions. A license for the code does not automatically grant permission to redistribute the dataset.
