import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os

# ✅ Load datasets safely (skip missing files)
datasets = {}
for name, file in {
    "banking": "synthetic_banking_data.csv",
    "loan": "loan_credit_analysis.csv",
    "fraud": "synthetic_fraud_detection.csv",
    "subscription": "subscription_spending_patterns.csv",
    "wealth": "synthetic_wealth_management.csv",
    "business": "business_financials.csv",
    "market": "market_insights.csv"
}.items():
    try:
        datasets[name] = pd.read_csv(file)
        print(f"✅ Loaded dataset: {name}")
    except FileNotFoundError:
        print(f"⚠️ Skipping missing dataset: {name}")
        datasets[name] = None  # Ensure missing datasets are handled gracefully

# ✅ Function to train models safely
def train_and_save_model(df, feature_cols, target_col, model_name, model, is_classification=True):
    """Trains an ML model safely, ignoring missing columns, and saves it."""
    if df is None or df.empty:
        print(f"⚠️ Skipping {model_name}: Dataset is empty or missing.")
        return

    # ✅ Ensure required columns exist
    existing_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"⚠️ {model_name}: Missing columns {missing_cols}. Training with available features: {existing_cols}")

    if not existing_cols:
        print(f"⚠️ Skipping {model_name}: No valid feature columns available.")
        return

    # ✅ Handle missing target column
    if target_col and target_col not in df.columns:
        print(f"⚠️ Skipping {model_name}: Target column '{target_col}' not found.")
        return

    X, y = df[existing_cols], df[target_col] if target_col else None

    # ✅ Convert categorical target variable to numbers
    if y is not None and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, f"{model_name}_label_encoder.pkl")  # ✅ Save encoder for decoding later

    # ✅ Convert continuous target variable to binary classification for Subscription Model
    if model_name == "subscription_model":
        y = (y > 0.5).astype(int)  # Convert to 0 or 1

    # ✅ Train and save model
    try:
        if y is not None:
            model.fit(X, y)
        else:
            model.fit(X)  # For unsupervised models like Isolation Forest

        joblib.dump(model, f"{model_name}.pkl")
        print(f"✅ Model trained & saved: {model_name}.pkl")
    except Exception as e:
        print(f"❌ Training failed for {model_name}: {e}")

# ✅ Special Handling for Fraud Model (Correct Features & Encoding)
def train_fraud_model():
    """Trains fraud model using correct dataset structure."""
    fraud_df = datasets.get("fraud")

    if fraud_df is None or fraud_df.empty:
        print(f"⚠️ Skipping fraud_model: Fraud dataset is empty or missing.")
        return

    # ✅ Correct column names
    fraud_features = ['Transaction Amount', 'Risk Score', 'Transaction Type', 'Location', 'Device Used']
    available_features = [col for col in fraud_features if col in fraud_df.columns]

    if not available_features:
        print(f"⚠️ Skipping fraud_model: No valid fraud detection features found.")
        return

    # ✅ Encode categorical features
    for col in ['Transaction Type', 'Location', 'Device Used']:
        if col in fraud_df.columns:
            label_encoder = LabelEncoder()
            fraud_df[col] = label_encoder.fit_transform(fraud_df[col])

    # ✅ Train fraud detection model
    X_fraud = fraud_df[available_features]
    y_fraud = fraud_df['Fraud Label'] if 'Fraud Label' in fraud_df.columns else None

    try:
        fraud_model = xgb.XGBClassifier(n_estimators=300, eval_metric='logloss')
        fraud_model.fit(X_fraud, y_fraud)
        joblib.dump(fraud_model, "fraud_model.pkl")
        print("✅ Fraud model trained & saved: fraud_model.pkl")
    except Exception as e:
        print(f"❌ Training failed for fraud_model: {e}")

# ✅ Train & Save Each Model Safely
train_and_save_model(datasets.get("loan"), ['age', 'income', 'credit_score', 'existing_loans'], 'loan_approval_status', "loan_model", xgb.XGBClassifier(n_estimators=300, eval_metric='logloss'))
train_fraud_model()  # ✅ Special handling for fraud model
train_and_save_model(datasets.get("subscription"), ['Monthly_Fee', 'Monthly_Spending'], 'Cancel_Probability', "subscription_model", LogisticRegression(), is_classification=True)
train_and_save_model(datasets.get("wealth"), ['Net_Worth', 'Annual_Income', 'Debt_to_Income_Ratio'], 'Risk_Appetite', "wealth_model", RandomForestClassifier())

print("✅ All ML models trained and saved successfully!")
