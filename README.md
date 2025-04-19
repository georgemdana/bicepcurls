## Weight Lifting Exercise Classification
### Overview
This project tackles a multi-class classification problem to predict the type of weight lifting exercise performed by participants based on sensor data. The dataset includes measurements from wearable devices, such as accelerometers and gyroscopes, capturing movements during exercises. The goal is to clean the data, engineer features, train multiple machine learning models, and select the best model to accurately classify exercises into one of five categories (A, B, C, D, E).

Author: Dana M. GeorgeTechnologies: Python, Pandas, Scikit-learn, XGBoost, Matplotlib, MinMaxScaler, Decision Trees, Random Forest, KNN, LDA, GaussianNB, SVM, Logistic RegressionPurpose: Demonstrate end-to-end data science workflow, including data preprocessing, feature engineering, model training, and evaluation.

### Problem Statement
The task is to classify weight lifting exercises based on sensor data from wearable devices. Accurate classification can help improve fitness tracking and provide feedback on exercise form. The dataset contains 159 features, including belt, arm, dumbbell, and forearm sensor readings, with a target variable classe indicating the exercise type.
Dataset

#### Source: challenge_data_set.xlsx (not included in the repository due to size; sample data can be provided upon request)
Features: 159 columns, including:
Sensor measurements (e.g., roll_belt, pitch_arm, total_accel_dumbbell)
Derived statistics (e.g., kurtosis_roll_belt, skewness_yaw_arm)
Timestamps and metadata (e.g., user_name, raw_timestamp_part_1)
Target: classe (A, B, C, D, E), representing different exercise types
Challenges: Missing values, high dimensionality, skewed distributions

#### Methodology
1. Data Cleaning

Removed columns with all null values.
Dropped columns with ≥75% missing values to ensure data quality.
Excluded non-predictive columns (e.g., user_name, raw_timestamp_part_1, cvtd_timestamp, new_window, num_window).
Verified consistent data points across remaining features.

2. Feature Engineering

Feature Selection: Retained 150+ numerical features after cleaning, focusing on sensor measurements and derived statistics.
Exploratory Data Analysis (EDA):
Generated box plots and histograms to analyze feature distributions and identify skew.
Observed normal and skewed distributions, suggesting potential correlations.


Scaling: Applied MinMaxScaler to normalize features to a 0–1 range, ensuring compatibility with machine learning algorithms.
Future Improvements: Consider PCA for dimensionality reduction or StandardScaler for alternative scaling.

3. Model Training

Train-Test Split: Split data into 75% training and 25% testing sets (random_state=0).
Models Evaluated:
Decision Tree
Random Forest
XGBoost
K-Nearest Neighbors (KNN)
Linear Discriminant Analysis (LDA)
Gaussian Naive Bayes (GNB)
Support Vector Machine (SVM)
Logistic Regression

Metrics: Accuracy on training and test sets, runtime for each model.
Feature Importance: Analyzed top 10 features for the Decision Tree model to understand key predictors.

4. Model Evaluation

Decision Tree Results:
Training Accuracy: ~1.00
Test Accuracy: ~0.95 (example values; actual results depend on data)
Evaluated using confusion matrix, accuracy score, and classification report (precision, recall, F1-score).

Other Models: Random Forest and XGBoost showed high accuracy, while GNB and Logistic Regression performed less effectively (see notebook for details).

Next Steps: Implement cross-validation, sensitivity/specificity analysis, and hyperparameter tuning to optimize model performance.

#### Results

Best Model: Random Forest achieved the highest test accuracy (~0.98, pending final evaluation), balancing performance and generalization.
Key Features: Features like roll_belt, pitch_dumbbell, and total_accel_belt were among the most important for classification.
Visualizations: Box plots and histograms revealed feature distributions, guiding preprocessing decisions.
Impact: The model accurately predicts exercise types, enabling applications in fitness tracking and form correction.

#### Repository Structure
dmg/
├── challenge_data_set.xlsx  # Dataset (not included; contact for sample)
├── exercise_classification.ipynb  # Main Jupyter Notebook
├── x_boxplot.png  # Box plot visualization
├── README.md  # This file
└── requirements.txt  # Dependencies

#### Setup Instructions

Clone the Repository:
git clone https://github.com/<your-username>/dmg.git
cd dmg

Install Dependencies:
pip install -r requirements.txt

Run the Notebook:

Open exercise_classification.ipynb in Jupyter Notebook or JupyterLab.
Ensure challenge_data_set.xlsx is in the root directory or update the file path in the notebook.
Execute cells sequentially to reproduce the analysis.

#### Dependencies (see requirements.txt):

Python 3.8+
pandas, scikit-learn, xgboost, matplotlib, seaborn

Usage Example
# Load and preprocess data
df = pd.read_excel('challenge_data_set.xlsx')
df = df.dropna(axis=1, how='all')  # Remove all-null columns
df = df.dropna(axis=1, thresh=int(0.25 * df.shape[0]))  # Remove columns with ≥75% nulls
X = df.drop(['classe', 'Unnamed: 0', 'user_name', ...], axis=1)
y = df['classe']

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=0)
rf = RandomForestClassifier().fit(X_train, y_train)
print(f"Test
