import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

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
# Encode Categorical
# -------------------------
cat_cols = ["family_history", "SMOKE", "FAVC"]
cat_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    cat_encoders[col] = le

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------
# Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": SVC(probability=True),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
    "XGBoost": XGBClassifier(
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
    )
}

results = {}

# -------------------------
# Evaluate Classical Models
# -------------------------
for name, model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("model", model)
    ])

    scores = cross_val_score(pipeline, X, y_encoded, cv=5)

    results[name] = {
        "Mean Accuracy": scores.mean(),
        "Std Dev": scores.std()
    }

# -------------------------
# TabNet (Special Handling)
# -------------------------
print("\nTraining TabNet...")

X_np = X.values
y_np = y_encoded

tabnet = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    verbose=0
)

pipeline_tabnet = Pipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
])

scores_tabnet = []

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X_np, y_np):

    X_train, X_test = X_np[train_idx], X_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]

    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tabnet.fit(X_train, y_train, max_epochs=100, patience=10)

    preds = tabnet.predict(X_test)

    scores_tabnet.append(accuracy_score(y_test, preds))

results["TabNet"] = {
    "Mean Accuracy": np.mean(scores_tabnet),
    "Std Dev": np.std(scores_tabnet)
}

# -------------------------
# Print Results (Paper Style)
# -------------------------
print("\n================ MODEL COMPARISON ================\n")

for model_name, metrics in results.items():
    print(f"{model_name}")
    print(f"Mean Accuracy : {metrics['Mean Accuracy']:.4f}")
    print(f"Std Dev       : {metrics['Std Dev']:.4f}")
    print("-----------------------------------")
