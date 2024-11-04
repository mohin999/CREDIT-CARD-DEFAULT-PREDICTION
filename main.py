
pip install gradio

import pandas as pd

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(url, header=1)
data.head()

# Basic info
data.info()
data.describe()

# Check for missing values
data.isnull().sum()

# Check for duplicates
data.duplicated().sum()

data = data.drop_duplicates()

import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(data['LIMIT_BAL'], kde=True)
plt.title('Distribution of Credit Limit')
plt.show()

sns.countplot(x='SEX',hue='default payment next month', data=data)
plt.title('Default Rate by Gender')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.histplot(data['AGE'], kde=True)
plt.title('Age Distribution')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load your data into X and y
X = data.drop(columns=['ID', 'default payment next month'])  # Features
y = data['default payment next month']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on X_train_resampled and transform both X_train_resampled and X_test
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# One-hot encode the categorical columns in the training and test set
X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# Align columns to ensure both train and test sets have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Scale the data
scaler = StandardScaler()
X_train_encoded_scaled = scaler.fit_transform(X_train_encoded)
X_test_encoded_scaled = scaler.transform(X_test_encoded)

# Initialize Logistic Regression with increased max_iter
logreg = LogisticRegression()
logreg.fit(X_train_encoded_scaled, y_train)

# Make predictions
y_pred = logreg.predict(X_test_encoded_scaled)

# Evaluate model using cross-validation
logreg_scores = cross_val_score(logreg, X_train_encoded_scaled, y_train, cv=5, scoring='accuracy')
print("Logistic Regression CV Accuracy: ", logreg_scores.mean())
print("Logistic Regression CV Standard Deviation: ", logreg_scores.std())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


# Separate features and target
X = data.drop(columns=['ID','default payment next month'])
y = data['default payment next month']


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42, class_weight="balanced")

# Train the model
rf_clf.fit(X_train, y_train)

# Evaluate model using cross-validation
rf_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
print("Random Forest CV Accuracy: ", rf_scores.mean())
print("Random Forest CV Standard Deviation: ", rf_scores.std())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Define the parameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Different solvers for logistic regression
}

# Initialize Logistic Regression and GridSearchCV
logreg_eval = LogisticRegression()
grid_search = GridSearchCV(logreg_eval, param_grid, cv=5, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train_encoded_scaled, y_train)

# Get the best model and its parameters
best_logreg = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate with cross-validation on the best model
cv_scores = cross_val_score(best_logreg, X_train_encoded_scaled, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Accuracy: {:.2f} ± {:.2f}".format(cv_scores.mean(), cv_scores.std()))

# Predict on the test set
y_pred = best_logreg.predict(X_test_encoded_scaled)

# Detailed Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define a reduced parameter grid for RandomizedSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

# Initialize Random Forest Classifier
rf_clf = RandomForestClassifier()

# Hyperparameter tuning with RandomizedSearchCV (100 iterations)
random_search = RandomizedSearchCV(rf_clf, param_grid, cv=5, scoring='f1', n_iter=20, random_state=42)
random_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_rf_clf = random_search.best_estimator_
print("Best Hyperparameters:", random_search.best_params_)

# Make predictions on the test set
y_pred = best_rf_clf.predict(X_test_scaled)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Evaluation Metrics on Test Set:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load and preprocess the dataset
data = pd.read_excel(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    header=1
)

# Define target column and select features for training
target_column = 'default payment next month'
selected_features = ['LIMIT_BAL', 'AGE', 'PAY_0', 'BILL_AMT1']
X = data[selected_features]
y = data[target_column]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train a Random Forest classifier
model_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model_rf.fit(X_resampled, y_resampled)

# Define prediction function
def predict_default(LIMIT_BAL, AGE, PAY_0, BILL_AMT1):
    input_data = scaler.transform([[LIMIT_BAL, AGE, PAY_0, BILL_AMT1]])
    prediction = model_rf.predict(input_data)
    return "Default" if prediction[0] == 1 else "No Default"

import gradio as gr

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_default,
    inputs=[
        gr.Number(label="Credit Limit (LIMIT_BAL)"),
        gr.Number(label="Age"),
        gr.Number(label="PAY_0"),
        gr.Number(label="BILL_AMT1")
    ],
    outputs="text",
    title="Credit Card Default Prediction",
    description="Predicts if a client will default based on credit limit, age, repayment status, and bill amount."
)

# Launch the Gradio app
interface.launch()
