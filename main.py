import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from collections import Counter

file_path = "heart_attack_prediction_dataset.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

print("\nData Info:")
print(df.info())

print("\nMissing values per column")
print(df.isnull().sum())

print("\nHeart Attack Risk distribution:")
print(df['Heart Attack Risk'].value_counts(normalize=True))


# --- Data Preprocessing ---
columns_to_drop = ["Patient ID", "Country", "Continent", "Hemisphere"]
df = df.drop(columns_to_drop, axis=1)

bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp_split[0], errors='coerce')
df["Diastolic_BP"] = pd.to_numeric(bp_split[1], errors='coerce')
df.drop("Blood Pressure", axis=1, inplace=True)

if df[["Systolic_BP", "Diastolic_BP"]].isnull().sum().any():
    df.dropna(subset=["Systolic_BP", "Diastolic_BP"], inplace=True)

df = pd.get_dummies(df, columns=['Sex', 'Diet'], drop_first=True)

X = df.drop("Heart Attack Risk", axis=1)
y = df["Heart Attack Risk"]
print(f"Number of features after preprocessing: {X.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")


# --- Model Training ---
model = LogisticRegression(solver="liblinear", random_state=42, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")


# --- Predictions ---
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


# --- Model Evaluation ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Class 1): {precision:.4f}")
print(f"Recall (Class 1): {recall:.4f}")
print(f"F1 Score (Class 1): {f1:.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Logistic Regression)")
plt.tight_layout()
plt.savefig("confusion_matrix_lr_improved.png")
plt.close()


# --- Feature Importance ---
coefficients = model.coef_[0]
features = X_scaled_df.columns
feature_importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nFeature Importance (coefficients):")
print(feature_importance)

plt.figure(figsize=(10, 6))
top_n = 20
sns.barplot(x="Coefficient", y="Feature", data=feature_importance.head(top_n), palette='viridis')
plt.title(f"Top {top_n} Feature Importances (Logistic Regression)")
plt.tight_layout()
plt.savefig("feature_importance_plot_lr_improved.png")
plt.close()

print("\nCode execution complete. Check generated plots.")