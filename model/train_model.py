import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline

os.makedirs("saved_model", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data/Obesity prediction.csv")

# -------------------------
# Feature Engineering
# -------------------------
df["Weight_to_Age"] = df["Weight"] / df["Age"]
df["Height_to_Age"] = df["Height"] / df["Age"]
df["Activity_Ratio"] = df["FAF"] / (df["TUE"] + 1)
df["Activity_Hydration"] = df["FAF"] * df["CH2O"]

features = [
    "Age", "Height", "Weight",
    "family_history", "SMOKE", "FAVC",
    "FAF", "TUE", "CH2O",
    "Weight_to_Age",
    "Height_to_Age",
    "Activity_Ratio",
    "Activity_Hydration"
]

target = "Obesity"

X = df[features].copy()
y = df[target]

# -------------------------
# Encode Categorical Features
# -------------------------
categorical_cols = ["family_history", "SMOKE", "FAVC"]
cat_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    cat_encoders[col] = le

# -------------------------
# Encode Target
# -------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -------------------------
# Build Pipeline
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        min_child_weight=2,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42
    ))
])

# -------------------------
# Train Pipeline
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nXGBoost Test Accuracy:", round(accuracy, 4))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------
# Confusion Matrix (Improved)
# -------------------------
cm = confusion_matrix(y_test, y_pred)

# Normalize (important for papers)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure()
plt.imshow(cm_normalized)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.colorbar()

class_names = label_encoder.classes_
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                 ha="center", va="center")

plt.tight_layout()
plt.savefig("figures/confusion_matrix.png")
plt.show()

# -------------------------
# Feature Importance (Paper Bonus)
# -------------------------
model = pipeline.named_steps["model"]

importances = model.feature_importances_

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance (XGBoost)")
plt.xticks(rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig("figures/feature_importance.png")
plt.show()

# -------------------------
# Cross Validation
# -------------------------
cv_scores = cross_val_score(pipeline, X, y_encoded, cv=5)
print("CV Accuracy:", round(cv_scores.mean(), 4))

# -------------------------
# Save Components
# -------------------------
joblib.dump(model, "saved_model/xgboost_model.pkl")
joblib.dump(pipeline.named_steps["scaler"], "saved_model/scaler.pkl")
joblib.dump(label_encoder, "saved_model/label_encoder.pkl")
joblib.dump(cat_encoders, "saved_model/categorical_encoders.pkl")

print("\nModel and preprocessing objects saved successfully!")
