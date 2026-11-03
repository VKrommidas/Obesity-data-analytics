# Data Preprocessing for Obesity Dataset
# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder , MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import time
import os
os.makedirs("plots", exist_ok=True)
print("File 'plots/' has been created or already existed.")

# Step 2: Load Dataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "Data1.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name="Φύλλο2", engine="openpyxl")

# Step 3: Convert numeric columns
#numeric_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
#for col in numeric_cols:
  # Replace commas with dots, remove stray characters (like spaces), then convert
  #df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' ', '').str.strip()
  #df[col] = pd.to_numeric(df[col], errors='coerce')


# Step 3: Convert and Clean Numeric Columns
numeric_cols = [
    'Age', 'Height', 'Weight',
    'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'
]

df_raw = df.copy()
raw_height = pd.to_numeric(
    df_raw['Height'],
    errors='coerce'
)
# Diagnostic: Test alternative height recovery
def recover_height_candidate(value):

    if pd.isna(value):
        return np.nan, "missing"

    value = float(value)

    # Already in centimetres
    if 100 <= value <= 250:
        return value, "already_cm"

    # Original height expressed in metres
    if 1 <= value <= 2.5:
        return value * 100, "metres_to_cm"

    # Decimal separator removed from values such as 1.8
    if 10 <= value < 25:
        return value * 10, "short_value_to_cm"

    # Large values with removed decimal separators
    attempts = 0

    while value > 250 and attempts < 10:
        value /= 10
        attempts += 1

    if 100 <= value <= 250:
        return value, f"divided_by_10^{attempts}"

    return np.nan, "unrecoverable"


height_results = raw_height.apply(recover_height_candidate)

candidate_height = height_results.apply(lambda x: x[0])
recovery_methods = height_results.apply(lambda x: x[1])

print("\n========== HEIGHT RECOVERY METHODS ==========")

print(recovery_methods.value_counts())

print("\nMean height by recovery method:")

print(
    pd.DataFrame({
        "Height": candidate_height,
        "Method": recovery_methods
    }).groupby("Method")["Height"].agg(
        ["count", "mean", "std", "min", "max"]
    )
)

print("\n==============================================")




for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.strip().str.replace(',', '.', regex=False),
        errors='coerce'
    )


def recover_decimal(value, min_value, max_value, column):

    if pd.isna(value):
        return np.nan

    value = float(value)

    if column == 'Height':

        # Already in centimetres
        if 100 <= value <= 250:
            pass

        # Originally expressed in metres
        elif 1 <= value <= 2.5:
            value *= 100

        # Decimal separator removed from values such as 1.8
        elif 10 <= value < 25:
            value *= 10

        # Large values with removed decimal separators
        else:
            attempts = 0

            while value > 250 and attempts < 10:
                value /= 10
                attempts += 1

    else:
        attempts = 0

        while value > max_value and attempts < 10:
            value /= 10
            attempts += 1

    if min_value <= value <= max_value:
        return value

    return np.nan


valid_ranges = {
    'Age': (10, 100),
    'Height': (100, 250),
    'Weight': (30, 250),
    'FCVC': (1, 3),
    'NCP': (1, 4),
    'CH2O': (1, 3),
    'FAF': (0, 3),
    'TUE': (0, 2)
}

for col, (minimum, maximum) in valid_ranges.items():
    df[col] = df[col].apply(
        lambda x: recover_decimal(x, minimum, maximum, col)
    )

# Diagnostic: Validate recovered heights using BMI
height_validation = df_raw[['Gender', 'NObeyesdad']].copy()

height_validation['Height'] = candidate_height
height_validation['Weight'] = df['Weight']

height_validation['BMI'] = (
    height_validation['Weight'] /
    (height_validation['Height'] / 100) ** 2
)

print("\n========== BMI VALIDATION BY OBESITY CATEGORY ==========")

print(
    height_validation.groupby('NObeyesdad')['BMI'].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    ).round(2)
)

print("\n========== HEIGHT VALIDATION BY GENDER ==========")

print(
    height_validation.groupby('Gender')['Height'].agg(
        ['count', 'mean', 'median', 'std', 'min', 'max']
    ).round(2)
)

print("\nOverall BMI statistics:")

print(height_validation['BMI'].describe().round(2))
print("========================================================")



print("\n========== RECOVERY DIAGNOSTICS ==========")

for col in numeric_cols:
    print(
        f"{col}: "
        f"{df[col].notna().sum()} valid / "
        f"{df[col].isna().sum()} invalid"
    )

print("\nExamples of records with invalid values:")
print(
    df[df[numeric_cols].isna().any(axis=1)]
    [['Age', 'Height', 'Weight', 'FCVC',
      'NCP', 'CH2O', 'FAF', 'TUE']]
    .head(20)
)

print("\n==========================================")



print("\nCleaning numeric data...")
print("Original records:", len(df_raw))
# Step 4: Basic Exploration
print("Preview of Data:")
print(df.head())
print("\n Info:")
print(df.info())
print("\n Description of Numeric Columns:")
print(df[numeric_cols].describe())

# Detect and Handle Missing Values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 5: Drop rows with any missing numeric data
#df_cleaned = df.dropna(subset=numeric_cols)
df_cleaned = df.dropna(subset=numeric_cols).copy()
# Preserve cleaned measurements before any scaling or normalization
df_cleaned_original = df_cleaned.copy()

print("\n========== CLEANING RESULTS ==========")
print("Original records:", len(df_raw))
print("Records after cleaning:", len(df_cleaned))
print("Removed records:", len(df_raw) - len(df_cleaned))

print("\nCorrected numeric statistics:")
print(df_cleaned[numeric_cols].describe())


# Step 6: Detect and remove extreme outliers (impossible values)
print("\nChecking suspicious height/weight values:")
print(df[(df['Height'] < 100) | (df['Height'] > 250)])  # height in cm
print(df[(df['Weight'] < 30) | (df['Weight'] > 200)])  # weight in kg

# Fix likely meters in Height
df_cleaned = df_cleaned.copy()
df_cleaned['Height'] = df_cleaned['Height'].apply(lambda x: x*100 if x < 3 else x)

# Drop again any extreme outliers
df_cleaned = df_cleaned[(df_cleaned['Height'] >= 100) & (df_cleaned['Height'] <= 250)]
df_cleaned = df_cleaned[(df_cleaned['Weight'] >= 30) & (df_cleaned['Weight'] <= 200)]

# Step 7: Encode Categorical Variables
categorical_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
le = LabelEncoder()
for col in categorical_cols:
    df_cleaned[col] = le.fit_transform(df_cleaned[col])

# Step 8: Normalize Numerical Variables
numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
scaler = StandardScaler()
df_cleaned[numerical_cols] = scaler.fit_transform(df_cleaned[numerical_cols])

# Step 9: Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_cleaned[numerical_cols])
plt.title("Boxplot of Normalized Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/boxplot_normalized_features.png")
plt.show()


# Histograms
df_cleaned[numeric_cols].hist(bins=15, figsize=(15, 10), layout=(3, 3))
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.savefig("plots/histograms_numerical_features.png")
plt.show()

# Countplot for Target Class
plt.figure(figsize=(10, 6))
sns.countplot(data=df_cleaned, x='NObeyesdad', order=df_cleaned['NObeyesdad'].value_counts().index)
plt.title("Distribution of Obesity Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/obesity_class_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# Dimensionality Reduction with PCA
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_cleaned[numeric_cols].values)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title("PCA of Normalized Numerical Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("plots/pca_projection.png")
plt.show()


# Step 10: Clustering - Feature Selection (diet-related)
cluster_features = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
X = df_cleaned[cluster_features]

# Step 10a: Evaluate the Optimal Number of K-Means Clusters
silhouette_scores = []

for k in range(2, 11):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels_test = kmeans_test.fit_predict(X)

    score = silhouette_score(X, labels_test)

    silhouette_scores.append(score)

    print(f"K = {k}, Silhouette Score = {score:.3f}")

# Plot Silhouette Scores

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    silhouette_scores,
    marker='o'
)

plt.title("K-Means Silhouette Score for Different Cluster Counts")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.tight_layout()
plt.savefig("plots/kmeans_silhouette_analysis.png", dpi=300)
plt.show()

# Quick check of selected features
print("\nUnique values per selected feature:")
print(df_cleaned[cluster_features].nunique())
print("\nSummary stats of selected features:")
print(df_cleaned[cluster_features].describe())

# Plot original distributions to check skew
plt.figure(figsize=(12, 6))
for i, col in enumerate(cluster_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(df[col], kde=True, bins=10)
    plt.title(f"Raw Distribution of {col}")
plt.tight_layout()
plt.show()

# Optional: Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("\nExplained variance by PCA components:", pca.explained_variance_ratio_)

# Step 11: K-Means Clustering
start = time.time()
#kmeans = KMeans(n_clusters=3, random_state=42)
kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)
kmeans_labels = kmeans.fit_predict(X)
kmeans_score = silhouette_score(X, kmeans_labels)
end = time.time()
print(f"K-Means Silhouette Score: {kmeans_score:.3f}, Time: {end - start:.2f}s")

print("K-Means cluster distribution:")
print(pd.Series(kmeans_labels).value_counts())

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=30)
plt.title("K-Means Clusters (K=2, PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("plots/kmeans_clusters.png")
plt.show()

# Compare average lifestyle characteristics across clusters

cluster_analysis = df_cleaned_original.copy()

cluster_analysis["KMeans_Cluster"] = kmeans_labels

print("\nK-Means Cluster Profiles (Original Values):")

print(
    cluster_analysis.groupby("KMeans_Cluster")[
        ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
    ].mean().round(3)
)

print("\nK-Means Cluster Profiles (Standardized Values):")

print(
    pd.DataFrame(
        X,
        columns=cluster_features
    )
    .assign(KMeans_Cluster=kmeans_labels)
    .groupby("KMeans_Cluster")[cluster_features]
    .mean()
    .round(3)
)

# Step 12: DBSCAN Clustering
start = time.time()
dbscan = DBSCAN(eps=0.6, min_samples=4)
dbscan_labels = dbscan.fit_predict(X)
end = time.time()

# Filter out noise points (-1) for silhouette score
mask = dbscan_labels != -1
if np.sum(mask) > 1:
    dbscan_score = silhouette_score(X[mask], dbscan_labels[mask])
    print(f"DBSCAN Silhouette Score: {dbscan_score:.3f}, Time: {end - start:.2f}s")
else:
    print("DBSCAN produced mostly noise. Try adjusting eps/min_samples.")

print("DBSCAN cluster distribution:")
print(pd.Series(dbscan_labels).value_counts())

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='plasma', s=30)
plt.title("DBSCAN Clusters (PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.tight_layout()
plt.savefig("plots/dbscan_clusters.png")
plt.show()

# Step 13: Visual comparison to real classes
# Clustering vs True Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_cleaned['NObeyesdad'], palette='Set2', s=40)
plt.title("PCA View Colored by Obesity Level")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/pca_real_labels.png")
plt.show()

# (Optional)Save the Cleaned Dataset (optional)
#df_cleaned.to_csv("cleaned_obesity_data.csv", index=False)

# (Optional) Save clustered dataset
#df_cleaned['kmeans_cluster'] = kmeans_labels
#df_cleaned['dbscan_cluster'] = dbscan_labels
#df_cleaned.to_csv("clustered_obesity_data.csv", index=False)



# Step 3a: Classification - Predict Obesity Category

# Step 1: Prepare original, unscaled classification data
# Align with df_cleaned in case any records were removed during cleaning.

classification_data = df_cleaned_original.loc[df_cleaned.index].copy()

X_cls = classification_data.drop(columns=['NObeyesdad'])
y_cls = classification_data['NObeyesdad']

# Model B: Exclude Height and Weight to investigate potential target leakage
X_cls_lifestyle = X_cls.drop(columns=['Height', 'Weight'])

print("\nClassification features:")
print(X_cls.columns.tolist())

print("\nTarget variable:")
print(y_cls.name)


# Step 2: Train/Test Split
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

X_train_lifestyle, X_test_lifestyle, y_train_lifestyle, y_test_lifestyle = train_test_split(
    X_cls_lifestyle,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

print("\nClass distribution in training set:")
print(y_train_cls.value_counts())

print("\nClass distribution in test set:")
print(y_test_cls.value_counts())


# Step 3: Define preprocessing

numerical_features_cls = [
    col for col in X_cls.columns
    if col in numerical_cols
]

categorical_features_cls = [
    col for col in X_cls.columns
    if col not in numerical_features_cls
]

def create_classification_preprocessor(numeric_features, categorical_features):

    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        sparse_threshold=0
    )


# Step 4: Train classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# Random Forest - Model A (all features)
rf_model = Pipeline([
    ('preprocessor', create_classification_preprocessor(
        numerical_features_cls,
        categorical_features_cls
    )),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_model.fit(X_train_cls, y_train_cls)
y_pred_rf = rf_model.predict(X_test_cls)


# Random Forest - Model B (without Height and Weight)
numerical_features_lifestyle = [
    col for col in numerical_features_cls
    if col not in ['Height', 'Weight']
]

categorical_features_lifestyle = categorical_features_cls.copy()

rf_lifestyle = Pipeline([
    ('preprocessor', create_classification_preprocessor(
        numerical_features_lifestyle,
        categorical_features_lifestyle
    )),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_lifestyle.fit(X_train_lifestyle, y_train_lifestyle)
y_pred_rf_lifestyle = rf_lifestyle.predict(X_test_lifestyle)


# Naive Bayes - Model A (all features)
nb_model = Pipeline([
    ('preprocessor', create_classification_preprocessor(
        numerical_features_cls,
        categorical_features_cls
    )),
    ('classifier', GaussianNB())
])

nb_model.fit(X_train_cls, y_train_cls)
y_pred_nb = nb_model.predict(X_test_cls)


# Step 5: Evaluate classifiers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

print("\nRandom Forest - Model A (With Height and Weight):")
print(classification_report(y_test_cls, y_pred_rf, zero_division=0))

print("\nRandom Forest - Model B (Without Height and Weight):")
print(classification_report(y_test_lifestyle, y_pred_rf_lifestyle, zero_division=0))

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test_cls, y_pred_nb, zero_division=0))


# Step 6: Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(
    confusion_matrix(y_test_cls, y_pred_rf, labels=rf_model.classes_),
    display_labels=rf_model.classes_
).plot(ax=axes[0], xticks_rotation=45)

axes[0].set_title("Random Forest Confusion Matrix")

ConfusionMatrixDisplay(
    confusion_matrix(y_test_cls, y_pred_nb, labels=nb_model.classes_),
    display_labels=nb_model.classes_
).plot(ax=axes[1], xticks_rotation=45)

axes[1].set_title("Naive Bayes Confusion Matrix")

plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=300)
plt.show()


# Step 7: ROC-AUC Curve for Random Forest (Multiclass)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Use the classes learned by the fitted Random Forest pipeline
classes = rf_model.classes_

y_test_bin = label_binarize(y_test_cls, classes=classes)

# Obtain probabilities directly from the fitted pipeline
y_score_rf = rf_model.predict_proba(X_test_cls)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(len(classes)):

    fpr[i], tpr[i], _ = roc_curve(
        y_test_bin[:, i],
        y_score_rf[:, i]
    )

    roc_auc[i] = auc(fpr[i], tpr[i])


# Plot ROC curves
plt.figure(figsize=(10, 8))

colors = plt.colormaps.get_cmap('tab10').resampled(len(classes))

for i in range(len(classes)):

    plt.plot(
        fpr[i],
        tpr[i],
        label=f"Class {classes[i]} (AUC = {roc_auc[i]:.2f})",
        color=colors(i)
    )

plt.plot([0, 1], [0, 1], 'k--', label='Random')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves per Obesity Class (Random Forest)')
plt.legend(loc="lower right", fontsize='small')

plt.tight_layout()
plt.savefig("plots/roc_auc_random_forest.png", dpi=300)
plt.show()

# Step 3b: Regression - Predict BMI with TensorFlow/Keras
# Step 1: Calculate BMI using corrected measurements
#df_bmi = df_cleaned.copy()
df_bmi = df_cleaned_original.copy()

df_bmi['BMI'] = (
    df_bmi['Weight'] / ((df_bmi['Height'] / 100) ** 2)
)

# Step 2: Define features and target
# Exclude Height and Weight to avoid direct target leakage

categorical_cols_bmi = [
    'Gender',
    'family_history_with_overweight',
    'FAVC',
    'CAEC',
    'SMOKE',
    'SCC',
    'CALC',
    'MTRANS'
]

numerical_cols_bmi = [
    'Age',
    'FCVC',
    'NCP',
    'CH2O',
    'FAF',
    'TUE'
]

feature_cols = numerical_cols_bmi + categorical_cols_bmi

X_reg = df_bmi[feature_cols]
y_reg = df_bmi['BMI']


print("\n========== CORRECTED BMI CHECK ==========")

print("Height range:",
      df_bmi['Height'].min(),
      "to",
      df_bmi['Height'].max())

print("Weight range:",
      df_bmi['Weight'].min(),
      "to",
      df_bmi['Weight'].max())

print("\nBMI statistics:")
print(df_bmi['BMI'].describe())

print("=========================================")

# Step 3: Split dataset before preprocessing
from sklearn.model_selection import train_test_split

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# Step 4: Preprocess numerical and categorical features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols_bmi),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols_bmi)
    ]
)

X_train_reg = preprocessor_reg.fit_transform(X_train_reg)
X_test_reg = preprocessor_reg.transform(X_test_reg)

# Convert to dense arrays for TensorFlow/Keras
X_train_reg = X_train_reg.astype(np.float32)
X_test_reg = X_test_reg.astype(np.float32)

# Convert target variables to NumPy arrays
y_train_reg = y_train_reg.to_numpy(dtype=np.float32)
y_test_reg = y_test_reg.to_numpy(dtype=np.float32)


# Regression data diagnostics

print("\n========== REGRESSION DATA DIAGNOSTICS ==========")

print("Training features shape:", X_train_reg.shape)
print("Testing features shape:", X_test_reg.shape)

print("Training features dtype:", X_train_reg.dtype)
print("Testing features dtype:", X_test_reg.dtype)

print("Training features - NaN:", np.isnan(X_train_reg).sum())
print("Training features - Inf:", np.isinf(X_train_reg).sum())

print("Testing features - NaN:", np.isnan(X_test_reg).sum())
print("Testing features - Inf:", np.isinf(X_test_reg).sum())

print("\nTraining features range:")
print("Minimum:", X_train_reg.min())
print("Maximum:", X_train_reg.max())

print("\nTraining BMI target:")
print("Minimum:", y_train_reg.min())
print("Maximum:", y_train_reg.max())
print("Mean:", y_train_reg.mean())
print("Standard deviation:", y_train_reg.std())

print("\nTesting BMI target:")
print("Minimum:", y_test_reg.min())
print("Maximum:", y_test_reg.max())
print("Mean:", y_test_reg.mean())
print("Standard deviation:", y_test_reg.std())

print("================================================")


# Step 3c: Regression Baseline Models
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Baseline 1: Predict the training-set mean
dummy_reg = DummyRegressor(strategy='mean')
dummy_reg.fit(X_train_reg, y_train_reg)

y_pred_dummy = dummy_reg.predict(X_test_reg)

# Baseline 2: Random Forest Regression
rf_reg = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_reg.fit(X_train_reg, y_train_reg)

y_pred_rf = rf_reg.predict(X_test_reg)

# Evaluate both baseline models
print("\n========== REGRESSION BASELINE RESULTS ==========")

for name, predictions in [
    ("Dummy Regressor", y_pred_dummy),
    ("Random Forest Regressor", y_pred_rf)
]:
    mae = mean_absolute_error(y_test_reg, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_reg, predictions))
    r2 = r2_score(y_test_reg, predictions)

    print(f"\n{name}:")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")

print("=================================================")

# Step 4: Build Feedforward Neural Network
import tensorflow as tf
from tensorflow.keras import layers, models , Input

# Define model
model_ff = models.Sequential([
    Input(shape=(X_train_reg.shape[1],)),         # Define input layer explicitly
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Linear output for regression
])
model_ff.compile(optimizer='adam', loss='mse', metrics=['mae'])

history_ff = model_ff.fit(X_train_reg, y_train_reg, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Step 5: Evaluate FF model
loss_ff, mae_ff = model_ff.evaluate(X_test_reg, y_test_reg,verbose=0)
y_pred_ff = model_ff.predict(X_test_reg,verbose=0).flatten()

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
rmse_ff = np.sqrt(mean_squared_error(y_test_reg, y_pred_ff))
mape_ff = mean_absolute_percentage_error(y_test_reg, y_pred_ff)
r2_ff = r2_score(y_test_reg, y_pred_ff)

print("\nFeedforward Model Evaluation:")
print(f"MAE: {mae_ff:.2f}, RMSE: {rmse_ff:.2f}, MAPE: {mape_ff:.2f}, R²: {r2_ff:.3f}")

# Plot first model training loss
plt.figure(figsize=(8, 5))
plt.plot(history_ff.history['val_loss'], label='val_loss')
plt.plot(history_ff.history['loss'], label='loss')
plt.title("Feedforward Model Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.savefig("plots/feedforward_training_curve.png")
plt.show()

# Step 6: Build Second Feedforward Neural Network
# Note: This is a placeholder - regression transfer learning in Keras is often customized
# For simplicity, we clone the FF model but label it as transfer model
# Step 6: Build Second Feedforward Neural Network

model_ff_2 = models.clone_model(model_ff)

model_ff_2.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history_ff_2 = model_ff_2.fit(
    X_train_reg,
    y_train_reg,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Step 7: Evaluate Second Feedforward Model

loss_ff_2, mae_ff_2 = model_ff_2.evaluate(
    X_test_reg,
    y_test_reg,
    verbose=0
)

y_pred_ff_2 = model_ff_2.predict(
    X_test_reg,
    verbose=0
).flatten()

rmse_ff_2 = np.sqrt(mean_squared_error(y_test_reg, y_pred_ff_2))
mape_ff_2 = mean_absolute_percentage_error(y_test_reg, y_pred_ff_2)
r2_ff_2 = r2_score(y_test_reg, y_pred_ff_2)

print("\nSecond Feedforward Model Evaluation:")
print(f"MAE:  {mae_ff_2:.2f}")
print(f"RMSE: {rmse_ff_2:.2f}")
print(f"MAPE: {mape_ff_2:.2f}")
print(f"R²:   {r2_ff_2:.3f}")

# Step 8: Consolidated Regression Model Comparison

regression_results = pd.DataFrame({
    'Model': [
        'Dummy Regressor',
        'Random Forest Regressor',
        'Feedforward Neural Network 1',
        'Feedforward Neural Network 2'
    ],
    'MAE': [
        mean_absolute_error(y_test_reg, y_pred_dummy),
        mean_absolute_error(y_test_reg, y_pred_rf),
        mean_absolute_error(y_test_reg, y_pred_ff),
        mean_absolute_error(y_test_reg, y_pred_ff_2)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test_reg, y_pred_dummy)),
        np.sqrt(mean_squared_error(y_test_reg, y_pred_rf)),
        np.sqrt(mean_squared_error(y_test_reg, y_pred_ff)),
        np.sqrt(mean_squared_error(y_test_reg, y_pred_ff_2))
    ],
    'R2': [
        r2_score(y_test_reg, y_pred_dummy),
        r2_score(y_test_reg, y_pred_rf),
        r2_score(y_test_reg, y_pred_ff),
        r2_score(y_test_reg, y_pred_ff_2)
    ]
})

# Sort models by R² (highest first)
regression_results = regression_results.sort_values(
    by='R2',
    ascending=False
).reset_index(drop=True)

print("\n========== REGRESSION MODEL COMPARISON ==========")
print(regression_results.round(3).to_string(index=False))
print("=================================================")

# Save regression comparison results
regression_results.to_csv(
    "plots/regression_model_comparison.csv",
    index=False
)
print("\nRegression comparison saved successfully.")


# Step 9: Visualize Regression Model Comparison

# Sort by R² for consistent presentation
plot_results = regression_results.sort_values(
    by='R2',
    ascending=False
)

# 1. MAE Comparison
plt.figure(figsize=(10, 6))

plt.bar(plot_results['Model'], plot_results['MAE'])

plt.title('Regression Models - MAE Comparison')
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('plots/regression_mae_comparison.png', dpi=300)
plt.show()


# 2. RMSE Comparison
plt.figure(figsize=(10, 6))

plt.bar(plot_results['Model'], plot_results['RMSE'])

plt.title('Regression Models - RMSE Comparison')
plt.xlabel('Model')
plt.ylabel('Root Mean Squared Error')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('plots/regression_rmse_comparison.png', dpi=300)
plt.show()


# 3. R² Comparison
plt.figure(figsize=(10, 6))

plt.bar(plot_results['Model'], plot_results['R2'])

plt.title('Regression Models - R² Comparison')
plt.xlabel('Model')
plt.ylabel('R² Score')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig('plots/regression_r2_comparison.png', dpi=300)
plt.show()


# Plot second model training loss
plt.figure(figsize=(8, 5))
plt.plot(history_ff_2.history['loss'], label='Training Loss')
plt.plot(history_ff_2.history['val_loss'], label='Validation Loss')
plt.title("Second Feedforward Model Training Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("plots/second_feedforward_training_curve.png")
plt.show()

# Export plots

import shutil

shutil.make_archive("plots", "zip", "plots")
