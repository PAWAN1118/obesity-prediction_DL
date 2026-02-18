import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

# -------------------------------
# Custom Focal Loss
# -------------------------------
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        return tf.reduce_mean(tf.reduce_sum(weight * cross_entropy, axis=1))
    return loss


# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("data/Obesity prediction.csv")

# -------------------------------
# Select ONLY 9 Features
# -------------------------------
features = [
    "Age",
    "Height",
    "Weight",
    "family_history",
    "SMOKE",
    "FAVC",
    "FAF",
    "TUE",
    "CH2O"
]

target_column = "Obesity"

X = df[features].copy()
y = df[target_column]

# -------------------------------
# Encode Target
# -------------------------------
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# -------------------------------
# Encode Categorical Columns
# -------------------------------
cat_cols = X.select_dtypes(include=['object']).columns
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# -------------------------------
# Scale Features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Save Preprocessing Objects
# -------------------------------
joblib.dump(scaler, "saved_model/scaler.pkl")
joblib.dump(target_encoder, "saved_model/label_encoder.pkl")
joblib.dump(encoders, "saved_model/categorical_encoders.pkl")

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# -------------------------------
# Build Model
# -------------------------------
input_layer = Input(shape=(X_train.shape[1],))

x = Dense(256, activation='relu')(input_layer)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Attention
attention = Dense(256, activation='softmax')(x)
x = Multiply()([x, attention])

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

output = Dense(y_train.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=focal_loss(gamma=2, alpha=0.25),
    metrics=['accuracy']
)

model.summary()

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# -------------------------------
# Save Model
# -------------------------------
model.save("saved_model/obesity_model.keras")

print("Training Completed ✅")
