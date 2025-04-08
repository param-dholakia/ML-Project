#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Set plot style
plt.style.use('seaborn')

# 1. Data Loading
print("Loading dataset...")
data = pd.read_csv('natural_disasters_dataset.csv')

# 2. Basic EDA
print("\nDataset Info:")
print(data.info())

print("\nFirst 5 rows:")
print(data.head())

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 3. Simple Visualization
# Disaster Type Distribution
plt.figure(figsize=(10, 6))
data['Disaster Type'].value_counts().plot(kind='bar')
plt.title('Distribution of Disaster Types')
plt.xlabel('Disaster Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Data Preprocessing
# Select relevant features based on your dataset
features = ['Year', 'Dis Mag Value', 'Dis Mag Scale', 'Latitude', 'Longitude', 'Disaster Type']
data_selected = data[features]

# Handle missing values
print("\nHandling missing values...")
data_selected = data_selected.replace('nan', np.nan)

# Impute numerical columns with mean
num_cols = ['Year', 'Dis Mag Value', 'Dis Mag Scale', 'Latitude', 'Longitude']
imputer = SimpleImputer(strategy='mean')
data_selected[num_cols] = imputer.fit_transform(data_selected[num_cols])

# Encode categorical target variable
le = LabelEncoder()
data_selected['Disaster Type'] = le.fit_transform(data_selected['Disaster Type'])

# Check for any remaining missing values
print("\nMissing values after imputation:")
print(data_selected.isnull().sum())

# 5. Prepare data for ML
X = data_selected.drop('Disaster Type', axis=1)
y = data_selected['Disaster Type']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Machine Learning Models
# Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

# KNN
print("Training KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_pred = knn_model.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, knn_pred)

# SVM
print("Training SVM...")
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)

# 7. Results
print("\nModel Performance:")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(f"KNN Accuracy: {knn_accuracy:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Detailed classification report for best model (Random Forest)
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=le.classes_))

# 8. Feature Importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# 9. Simple comparison plot
models = ['Random Forest', 'KNN', 'SVM']
accuracies = [rf_accuracy, knn_accuracy, svm_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies)
plt.title('Model Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
plt.tight_layout()
plt.show()