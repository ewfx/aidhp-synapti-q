import pandas as pd
import xgboost as xgb
import requests
from sklearn.metrics import accuracy_score, classification_report

# ✅ Load dataset
df = pd.read_csv("customer_engagement.csv")

# ✅ Display first few rows
print("📊 Dataset Overview:")
print(df.head())

# ✅ Check for missing values
print("\n❌ Missing Values:")
print(df.isnull().sum())

# ✅ Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# ✅ Check class distribution of churned customers
print("\n🔄 Churn Distribution:")
print(df["Churned"].value_counts(normalize=True))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Convert categorical column "Subscription_Status" into numerical values
df["Subscription_Status"] = df["Subscription_Status"].map({"Active": 0, "Cancelled": 1})

# ✅ Define Features (X) and Target (y)
X = df.drop(columns=["Customer_ID", "Churned"])  # Drop ID, keep relevant features
y = df["Churned"]  # Target variable

# ✅ Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🔹 Training Data Shape:", X_train.shape)
print("🔹 Testing Data Shape:", X_test.shape)



# ✅ Train XGBoost Model for Churn Prediction
churn_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
churn_model.fit(X_train, y_train)

# ✅ Define Function to Get LLM Insights (Sentiment & Chat Behavior)
def get_llm_churn_risk(customer_id):
    """
    Calls LLM API (Gemini API) to get sentiment & behavior analysis for a customer.
    """
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    API_KEY = "AIzaSyBWOJIzIMzkxni-ZL-4vcME0mwlirQ-BzI"

    prompt = f"""
    Analyze customer behavior and sentiment based on past chatbot interactions, support conversations, and spending habits.
    Given a customer with ID: {customer_id}, predict their likelihood of churn.
    
    Return a JSON response with:
    - "sentiment_score": Between -1 (very negative) to 1 (very positive)
    - "churn_risk": "High", "Medium", or "Low"
    """

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    response = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)

    if response.status_code == 200:
        result = response.json()
        try:
            churn_risk = result["candidates"][0]["content"]["parts"][0]["text"]
            return churn_risk
        except (IndexError, KeyError):
            return "LLM Analysis Failed"
    else:
        return "LLM API Error"

# ✅ Function to Combine Predictions (XGBoost + LLM)
def hybrid_churn_prediction(customer_id, customer_features):
    """
    Uses both XGBoost and LLM insights to make a final churn prediction.
    """
    # Get XGBoost Prediction
    xgb_pred = churn_model.predict([customer_features])[0]

    # Get LLM Prediction
    llm_risk = get_llm_churn_risk(customer_id)

    # Combine Both (Weighted Approach)
    if xgb_pred == 1 or "High" in llm_risk:
        final_risk = "High"
    elif "Medium" in llm_risk:
        final_risk = "Medium"
    else:
        final_risk = "Low"

    return {"customer_id": customer_id, "churn_risk": final_risk, "llm_analysis": llm_risk}

import joblib

# ✅ Save the Trained Model
joblib.dump(churn_model, "churn_model.pkl")
print("\n✅ Model saved as churn_model.pkl")


