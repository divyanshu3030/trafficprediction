

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("✅ Libraries imported successfully!")

# --- Global Variables for Data Cleaning ---
categorical_cols_for_imputation = [
    'State Name', 'Vehicle Type Involved', 'Weather Conditions', 'Road Condition',
    'Traffic Control Presence', 'Driver Gender', 'Alcohol Involvement'
]
numerical_cols = ['Speed Limit (km/h)', 'Driver Age']


# ============================================================
# Step 1 & 2: Load Data, Select Columns, and Robust Cleaning
# ============================================================
try:
    df = pd.read_csv("accident_prediction_india.csv")
    print("✅ Dataset loaded successfully!")
    print(f"Initial row count: {len(df)}")
except FileNotFoundError:
    print("❌ ERROR: File 'accident_prediction_india.csv' not found. Please ensure the file is uploaded.")
    exit()

selected_cols = numerical_cols + ['Accident Severity'] + categorical_cols_for_imputation
df = df[selected_cols]

# 1. Fill Numerical Missing Values with the Median
for col in numerical_cols:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# 2. Fill Categorical Missing Values with the Mode
for col in categorical_cols_for_imputation:
    # Convert to string before finding mode to handle mixed types
    df[col] = df[col].astype(str)
    mode_val = df[col].mode()[0]
    df[col].fillna(mode_val, inplace=True)

print(f"Row count after Imputation: {len(df)}")


# ============================================================
# Step 3: Convert Accident Severity to Binary (0 = No, 1 = Yes)
# ============================================================
# 1. Drop rows where the TARGET is completely missing (NaN/None)
df.dropna(subset=['Accident Severity'], inplace=True)

# 2. Robustly map text categories (like Fatal, Serious, Minor) to numbers.
if df['Accident Severity'].dtype == 'object':
    print("Detected 'Accident Severity' as a text column. Mapping to numeric severity levels.")
    # Assuming text values map to severity (higher number = more severe)
    severity_map = {
        'Fatal': 2, 'Serious': 1, 'Minor': 0, 'None': 0,
        'fatal': 2, 'serious': 1, 'minor': 0, 'none': 0,
    }
    df['Accident Severity'] = df['Accident Severity'].astype(str).str.strip().map(severity_map)

# 3. Convert to numeric and drop any row where conversion still fails
df['Accident Severity'] = pd.to_numeric(df['Accident Severity'], errors='coerce')
df.dropna(subset=['Accident Severity'], inplace=True)

if len(df) == 0:
    print("\n🚨 CRITICAL ERROR: DataFrame is empty after cleaning the target variable. Cannot proceed.")
    exit()

# 4. Final conversion to binary: (Accident Severity > 0) -> 1 (Yes, accident/serious), 0 (No/Minor)
df['Accident Severity'] = (df['Accident Severity'] > 0).astype(int)
print(f"Row count after final target cleaning: {len(df)}")


# ============================================================
# Step 4 & 5: Encode, Split, and Scale Data
# ============================================================
label_encoders = {}
for col in categorical_cols_for_imputation:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    # Fit the encoder on the *original* string values
    le.fit(df[col])
    df[col] = le.transform(df[col])
    label_encoders[col] = le

X = df.drop('Accident Severity', axis=1)
y = df['Accident Severity']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data prepared and scaled successfully!")

# ============================================================
# Step 6, 7, 8: Model Training and Saving (Unchanged)
# (Results will be printed here)
# ============================================================

# Random Forest Training...
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
print("\n" + "="*50)
print("🔹 Random Forest Results:")
print("="*50)
print(classification_report(y_test, y_pred_rf))
print("ROC AUC:", roc_auc_score(y_test, y_prob_rf))

# LSTM Training...
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
model = Sequential([
    LSTM(32, input_shape=(1, X_train_scaled.shape[1])),
    Dropout(0.3), Dense(16, activation='relu'), Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining LSTM Model...")
model.fit(X_train_lstm, y_train, epochs=10, batch_size=4, validation_split=0.2, verbose=0)
loss, acc = model.evaluate(X_test_lstm, y_test, verbose=0)
y_pred_prob = model.predict(X_test_lstm, verbose=0).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
print("\n" + "="*50)
print("🔹 LSTM Results:")
print("="*50)
print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))

# Saving Models...
model.save("lstm_model.h5")
joblib.dump(rf, "random_forest.pkl")
joblib.dump(scaler, "scaler.save")
joblib.dump(label_encoders, "label_encoders.pkl")
print("\n✅ All Models and Encoders Saved Successfully!")


# ============================================================
# Step 9: Interactive Prediction (Accepts TEXT/Numbers as is)
# ============================================================
print("\n" + "="*50)
print("🚦 Interactive Prediction Mode (Use TEXT/Numbers as in your dataset)")
print("="*50)

# Display Top 5 Options for Categorical Inputs (User Guide)
print("\n--- Input Reference Values (Top 5) ---")
for col in categorical_cols_for_imputation:
    # Use the original string values for reference
    top_values = df[col].astype(str).value_counts().head(5).index.tolist()
    print(f"**{col}**: {', '.join(top_values)}")
print("--------------------------------------")


user_input = {}
input_features = X.columns.tolist()

for feature in input_features:
    while True:
        try:
            # Handle Categorical Features (Accepts Text)
            if feature in label_encoders:
                prompt_text = f"Enter {feature} (Text, e.g., {df[feature].astype(str).mode()[0]}): "
                user_val = input(prompt_text).strip()
                user_input[feature] = user_val
                break

            # Handle Numerical Features (Accepts Number)
            elif feature in numerical_cols:
                avg_val = int(df[feature].mean())
                prompt_text = f"Enter {feature} (Number, e.g., {avg_val}): "
                user_val = float(input(prompt_text))
                user_input[feature] = user_val
                break
        except ValueError:
            print("Invalid input. Please enter a valid number for this field.")
        except EOFError:
            # Fallback for input environment issues
            print("\nInput cancelled. Using default values for remaining fields.")
            break

# Prepare data for prediction
input_data = []
for col in X.columns:
    if col in label_encoders:
        # **This is the key fix:** The code now transforms the TEXT input to the required number
        try:
            # Transform the user's TEXT input to the corresponding encoded number
            encoded_value = label_encoders[col].transform([str(user_input[col])])[0]
        except ValueError:
            print(f"Warning: '{user_input[col]}' not seen in training for '{col}'. Defaulting to label 0.")
            encoded_value = 0 # Fallback for unseen category
        input_data.append(encoded_value)
    else:
        # Append numerical input directly
        input_data.append(user_input[col])

sample = np.array([input_data])
sample_scaled = scaler.transform(sample)

# Random Forest Prediction
prob_rf = rf.predict_proba(sample_scaled)[0][1]

# LSTM Prediction
sample_lstm = sample_scaled.reshape(1, 1, len(X.columns))
prob_lstm = model.predict(sample_lstm, verbose=0)[0][0]

print("\n" + "="*50)
print("✨ Prediction Results")
print("="*50)
print(f"🚦 User Input Processed: {user_input}")
print(f"🟢 Random Forest Predicted Accident Probability: {prob_rf:.2f}")
print(f"🔵 LSTM Predicted Accident Probability: {prob_lstm:.2f}")

# Final Conclusion
avg_prob = (prob_rf + prob_lstm) / 2
threshold = 0.55
if avg_prob > threshold:
    print(f"\n⚠️ **High Risk of Accident** (Average Probability: {avg_prob:.2f}). Please drive with caution!")
else:
    print(f"\n✅ **Low Risk of Accident** (Average Probability: {avg_prob:.2f}). Safe journey!")

print("="*50)