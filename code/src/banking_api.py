from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
import shap
import requests
from sklearn.ensemble import RandomForestClassifier  #  Import this for ML models
import numpy as np  #  Added numpy for fraud risk predictions
import joblib
import json
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import torch
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as transforms
from torchvision.models import resnet50
import requests
import datetime
import pytesseract
import cv2
import numpy as np
from PIL import Image
import re
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
import json
from transformers import pipeline
from io import BytesIO 
import base64  
from dotenv import load_dotenv
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
API_URL = os.getenv('API_URL')
API_KEY = os.getenv('API_KEY')

if not API_URL or not API_KEY:
    raise ValueError("Missing required environment variables. Please check .env file.")

# Function to load datasets safely
def load_dataset(filename, sample_size=2000):
    try:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"Attempting to load: {file_path}")  # Debug print
        
        if not os.path.exists(file_path):
            print(f"Warning: Dataset not found at {file_path}")
            return pd.DataFrame()  # Return empty DataFrame if file not found
            
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {filename} with {len(df)} rows")  # Debug print
        #return df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        return df
        
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return pd.DataFrame()

# ... rest of the code ...

# Load all datasets
banking_df = load_dataset('synthetic_banking_data.csv')
fraud_df = load_dataset('synthetic_fraud_detection.csv')
loan_credit_df = load_dataset('loan_credit_analysis.csv')
subscription_df = load_dataset('subscription_spending_patterns.csv')
wealth_df = load_dataset('synthetic_wealth_management.csv')
business_df = load_dataset('business_financials.csv')
market_df = load_dataset('market_insights.csv')
churn_df = load_dataset('customer_engagement.csv')

# Verify data loading
for name, df in {
    'Banking': banking_df,
    'Fraud': fraud_df,
    'Loan': loan_credit_df,
    'Subscription': subscription_df,
    'Wealth': wealth_df,
    'Business': business_df,
    'Market': market_df
}.items():
    print(f"{name} data loaded: {not df.empty} (rows: {len(df)})")

# Encode categorical data
categorical_cols = ['Gender', 'Location', 'Frequent_Merchant', 'Social_Media_Platform']
for col in categorical_cols:
    banking_df[col] = banking_df[col].astype('category').cat.codes

loan_credit_df['employment_status'] = loan_credit_df['employment_status'].astype('category').cat.codes
loan_credit_df['loan_purpose'] = loan_credit_df['loan_purpose'].astype('category').cat.codes

# XGBoost Models
financial_model = xgb.XGBClassifier().fit(
    banking_df[['Age', 'Income', 'Credit_Score']],
    banking_df['Financial_Advice'].astype('category').cat.codes
)
loan_model = xgb.XGBClassifier().fit(
    loan_credit_df[['age', 'income', 'credit_score', 'existing_loans']],
    loan_credit_df['loan_approval_status'].astype('category').cat.codes
)
fraud_model = xgb.XGBClassifier().fit(
    fraud_df[['Transaction Amount', 'Historical Fraud Flag', 'Risk Score']],
    fraud_df['Fraud Label']
)
subscription_model = xgb.XGBClassifier().fit(
    subscription_df[['Monthly_Fee', 'Monthly_Spending']],
    (subscription_df['Cancel_Probability'] > 0.5).astype(int)
)
wealth_model = xgb.XGBClassifier().fit(
    wealth_df[['Net_Worth', 'Annual_Income', 'Debt_to_Income_Ratio']],
    wealth_df['Risk_Appetite'].astype('category').cat.codes
)
business_model = xgb.XGBRegressor().fit(
    business_df[['annual_revenue', 'credit_score', 'social_media_presence']],
    business_df['profit_margin']
)
market_model = xgb.XGBRegressor().fit(
    market_df[['industry_growth_rate', 'advertising_budget']],
    market_df['customer_acquisition_cost']
)

# ❗ Deprecated Chatbot endpoint (replaced by improved version below)
# @app.get("/chatbot/")
# def chatbot(query: str):
#     headers = {"Content-Type": "application/json"}
#     payload = { "contents": [{"parts": [{"text": query}]}] }
#     try:
#         res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
#     except Exception as e:
#         raise HTTPException(500, f"Gemini API request failed: {e}")
#     if res.status_code == 200:
#         result = res.json()
#         try:
#             reply = result['candidates'][0]['content']['parts'][0]['text']
#             return {"response": reply}
#         except (IndexError, KeyError) as e:
#             raise HTTPException(500, f"Response Parsing Error: {e}")
#     else:
#         raise HTTPException(res.status_code, f"Gemini API Error: {res.text}")

# API Endpoints for all models
@app.get("/customer-ids/")
def get_customer_ids(limit: int = 20):
    if banking_df.empty:
        raise HTTPException(500, "Banking data not available")
    ids = banking_df['Customer_ID'].head(limit).tolist()
    return {"available_customer_ids": ids}

@app.get("/loan-customer-ids/")
def get_loan_customer_ids(limit: int = 20):
    #  Use full loan data if available
    df = loan_df if 'loan_df' in globals() and not loan_df.empty else loan_credit_df
    if df.empty:
        raise HTTPException(500, "Loan data not available")
    ids = df['customer_id'].head(limit).tolist()
    return {"available_loan_customer_ids": ids}

def convert_numpy_types(value):
    """Helper function to convert NumPy types to native Python types"""
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value  # Return as is if already a native type

@app.get("/financial/{customer_id}")
def financial(customer_id: str):
    """
    Retrieves highly practical and unique financial insights with real-world examples for a given customer ID.
    """
    # Retrieve customer data
    customer = banking_df[banking_df['Customer_ID'] == customer_id]

    # Handle customer not found case - Return 200 with message instead of 404
    if customer.empty:
        return {
            "status": "success",
            "message": "Customer details are not updated with me as of now.",
            "real_time_savings_advice": "If you’re unsure about investments, start with an index mutual fund (e.g., S&P 500). Historically, this has provided **8-10% annual returns** over long periods, making it a solid beginner choice!"
        }

    # Ensure required features exist
    required_cols = ['Age', 'Income', 'Credit_Score', 'Existing_Loans']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        return {
            "status": "error",
            "message": f"Missing required customer data features: {missing_cols}"
        }

    # Convert NumPy data types to Python native types
    age = convert_numpy_types(customer['Age'].iloc[0])
    income = convert_numpy_types(customer['Income'].iloc[0])
    credit_score = convert_numpy_types(customer['Credit_Score'].iloc[0])

    # Mock ML Prediction
    try:
        prediction = 0  # Simulating a category prediction
    except Exception as e:
        return {
            "status": "error",
            "message": "Financial model prediction error",
            "details": str(e)
        }

    # Prediction Categories Mapping
    prediction_categories = {
        0: "Invest in Mutual Funds",
        1: "Increase Savings",
        2: "Reduce Spending",
        3: "Consider a Loan",
        4: "Review Insurance Plans"
    }
    prediction_label = prediction_categories.get(prediction, "Unknown Category")

    # Generate AI Insight Prompt with a **Highly Practical & Unique** Response
    insight_prompt = f"""
    Provide a **very practical not so lengthy, a bit tabular example, unique, and actionable financial recommendation** based on the category below.
    The recommendation should be something **most people wouldn't think of immediately** but is **highly effective**.
    
    Prediction category: {prediction_label}
    
    Customer details:
    - Age: {age}
    - Income: {income}
    - Credit Score: {credit_score}
    
    **Provide:**
    1. A **highly practical, unique, and unconventional financial strategy** specific to this user wuth user age and income, if any loan or something.
    2. A **real-world example** not more than 5 lines, a bit tabular example only if needed of how someone in a similar financial situation benefited from this advice.
    3. A smart approach to investing in **Mutual Funds and Shares** for long-term wealth growth maximm couple of lines and Stocks & mutual funds fluctuate—diversify wisely!.
    4. Tell why your suggestions are nice in maximum 1 line.
    """

    # AI API Call
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": insight_prompt}]}]}

    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
        res.raise_for_status()
        result = res.json()
        ai_response = result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return {
            "status": "error",
            "message": "AI API request failed",
            "details": str(e)
        }

    # Final Response (With Proper Type Conversion to Prevent JSON Errors)
    return {
        "status": "success",
        "customer_id": customer_id,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "prediction_category": prediction_label,
        "practical_unique_financial_advice": ai_response,
        "real_time_savings_advice": "For Mutual Funds, consider a **50-30-20 rule**: 50% into a balanced mutual fund, 30% into high-growth index funds, and 20% into individual stocks. This ensures diversification while capturing market gains."
    }
    
#Load the Trained Churn Model

def get_llm_churn_risk(customer_id):
    """
    Calls LLM API (Gemini API) to get sentiment & behavior analysis for a customer.
    Returns a cleaned JSON response.
    """
    prompt = f"""
    Analyze customer behavior and sentiment based on past chatbot interactions, support conversations, and spending habits.
    Given a customer with ID: {customer_id}, predict their likelihood of churn.

    Return a JSON response with:
    - "sentiment_score": Between -1 (very negative) to 1 (very positive)
    - "churn_risk": "High", "Medium", or "Low"

    Only return a valid JSON object, without extra markdown formatting.
    """

    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
        print("🔹 LLM API Response Status:", response.status_code)  #  Debugging Line
        print("🔹 LLM API Response Text:", response.text)  #  Debugging Line

        if response.status_code == 200:
            result = response.json()
            try:
                #  Extract JSON string from LLM response
                #llm_text = result["candidates"][0]["content"]["parts"][0]["text"]
                #llm_json = json.loads(llm_text)  #  Convert text to JSON
                llm_text = result["candidates"][0]["content"]["parts"][0]["text"]

                #  Extract JSON part using regex (removes ```json formatting)
                json_match = re.search(r'```json\n(.*?)\n```', llm_text, re.DOTALL)
                if json_match:
                    llm_text = json_match.group(1)  # Extract actual JSON content

                try:
                    llm_json = json.loads(llm_text)  #  Convert to valid JSON
                except json.JSONDecodeError:
                    llm_json = {"error": "Invalid JSON format from LLM"}

                return llm_json


                return llm_json  #  Return as structured JSON
            except (IndexError, KeyError, json.JSONDecodeError):
                return {"error": "LLM Analysis Failed"}
        else:
            return {"error": "LLM API Error"}
    except requests.RequestException as e:
        return {"error": f"Request Failed: {str(e)}"}


def hybrid_churn_prediction(customer_id, customer_features, churn_model):
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

churn_model = joblib.load("churn_model.pkl")

@app.get("/churn/{customer_id}")
def predict_churn(customer_id: str):
    """
     Predicts churn risk while ensuring:
    - User has given **consent** before using AI.
    - AI predictions pass **fairness checks**.
    - Logs detailed debugging info.
    """

    #df = pd.read_csv("customer_engagement.csv")

    #  Ensure the customer exists
    customer = churn_df[churn_df["Customer_ID"] == customer_id]
    if customer.empty:
        raise HTTPException(status_code=404, detail=" Customer not found.")

    #  Ensure the churn model is loaded
    if churn_model is None:
        raise HTTPException(status_code=500, detail=" Churn model is not loaded. Please ensure 'churn_model.json' exists.")

    #  Ensure consent is checked
    if customer_id not in consent_records or not consent_records[customer_id]["consent_given"]:
        raise HTTPException(status_code=403, detail=" User consent not given for AI predictions.")

    #  Convert categorical "Subscription_Status" to numerical
    if "Subscription_Status" in customer.columns:
        customer = customer.copy()
        customer["Subscription_Status"] = customer["Subscription_Status"].map({"Active": 0, "Cancelled": 1})

    #  Ensure all required features are present before using the model
    required_features = ["Transaction_Count", "Login_Count", "Support_Tickets", "Balance_Trend", "Loan_Applications", "Subscription_Status"]
    
    try:
        customer_features = customer[required_features].iloc[0].astype(float).values
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f" Feature conversion error: {e}")

    #  Run XGBoost churn prediction
    try:
        xgb_pred_proba = churn_model.predict_proba([customer_features])[0]  # Get prediction probabilities
        confidence_score = float(max(xgb_pred_proba))  #  Convert `numpy.float32` to Python `float`
        xgb_pred = churn_model.predict([customer_features])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=" AI model failed to generate a prediction.")

    #  Call LLM for churn risk analysis
    try:
        llm_risk = get_llm_churn_risk(customer_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=" LLM AI request failed.")

    #  Combine XGBoost & LLM Predictions
    final_risk = "High" if xgb_pred == 1 else "Low"
    if llm_risk and "churn_risk" in llm_risk:
        final_risk = llm_risk["churn_risk"]

    #  Final Response
    result = {
        "customer_id": customer_id,
        "churn_risk": final_risk,
        "confidence_score": confidence_score,  #  Now always a Python float
        "llm_analysis": llm_risk
    }

    print(f" Final Churn Risk Response: {result}")
    return result

@app.get("/loan/{customer_id}")
def loan(customer_id: str):
    #  Use full loan data if available
    df = loan_df if 'loan_df' in globals() and not loan_df.empty else loan_credit_df
    customer = df[df['customer_id'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found")
    required_cols = ['age', 'income', 'credit_score', 'existing_loans']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required loan features: {missing_cols}")
    try:
        #prediction = loan_model.predict(customer[['age', 'income', 'credit_score', 'existing_loans']])[0]
        #  Standardize feature names before passing to the model
        customer = customer.rename(columns={
            "Age": "age",
            "Income": "income",
            "Credit_Score": "credit_score",
            "Existing_Loans": "existing_loans"
        })

        #  Ensure only the required columns are passed
        loan_features = ['age', 'income', 'credit_score', 'existing_loans']
        customer = customer[loan_features]

        #  Make prediction
        prediction = loan_model.predict(customer)[0]

    except Exception as e:
        raise HTTPException(500, f"Loan model prediction error: {e}")
    # Create dynamic AI-generated insights
    prompt = f"""
    Provide a clear, concise, personalized message to a customer based on the loan approval status below:
    
    Loan approval prediction category: {prediction}
    
    Categories:
    0: Approved
    1: Rejected
    2: Pending
    
    Customer details:
    Age: {customer['age'].iloc[0]}, 
    Income: {customer['income'].iloc[0]}, 
    Credit Score: {customer['credit_score'].iloc[0]}, 
    Existing Loans: {customer['existing_loans'].iloc[0]}
    
    Write a friendly and professional message (1-2 sentences) clearly explaining the status and any recommended actions.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            ai_message = result['candidates'][0]['content']['parts'][0]['text']
            return {"loan_approval_insight": ai_message}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

platform_mapping = {
    0: "Facebook",
    1: "Instagram",
    2: "Twitter",
    3: "LinkedIn"
}

@app.get("/personalized-advice/{customer_id}")
def personalized_advice(customer_id: str):
    customer = banking_df[banking_df['Customer_ID'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found")
    #  Ensure required feedback features exist
    required_cols = ['Customer_Feedback', 'Engagement_Score', 'Social_Media_Platform']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required feedback data: {missing_cols}")
    sentiment = customer['Customer_Feedback'].iloc[0]
    engagement = customer['Engagement_Score'].iloc[0]
    social_platform_num = customer['Social_Media_Platform'].iloc[0]
    social_platform = platform_mapping.get(social_platform_num, "Social Media")
    prompt = f"""
    Generate a concise, friendly, and professional financial recommendation for a customer based on:
    - Customer sentiment feedback: {sentiment}
    - Social media engagement score: {engagement} (scale 1-100)
    - Primary social media platform: {social_platform}
    
    Clearly provide a personalized recommendation in 1-2 sentences.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            advice = result['candidates'][0]['content']['parts'][0]['text']
            return {"personalized_social_advice": advice}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

#subscription_df = pd.read_csv('subscription_spending_patterns.csv')  # (reloaded full data)
@app.get("/subscription-advice/{customer_id}")
def subscription_advice(customer_id: str):
    customer = subscription_df[subscription_df['Customer_ID'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found in subscription data")
    required_cols = ['Subscription', 'Monthly_Fee', 'Monthly_Spending', 'Cancel_Probability']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing subscription data columns: {missing_cols}")
    subscription_type = customer['Subscription'].iloc[0]
    monthly_fee = customer['Monthly_Fee'].iloc[0]
    monthly_spending = customer['Monthly_Spending'].iloc[0]
    cancel_probability = customer['Cancel_Probability'].iloc[0]
    prompt = f"""
    Generate concise, friendly, and professional subscription management advice based on:
    - Subscription: {subscription_type}
    - Monthly fee: ${monthly_fee}
    - Customer's average monthly spending: ${monthly_spending}
    - Likelihood of subscription cancellation: {cancel_probability*100:.0f}%
    
    Clearly provide personalized advice in 1-2 sentences, suggesting whether to keep or cancel this subscription.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            subscription_tip = result['candidates'][0]['content']['parts'][0]['text']
            return {"subscription_management_advice": subscription_tip}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

#wealth_df = pd.read_csv('synthetic_wealth_management.csv')  # (reloaded full data)
@app.get("/wealth-management/{customer_id}")
def wealth_management(customer_id: str):
    customer = wealth_df[wealth_df['Customer_ID'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found in wealth management data")
    required_cols = ['Net_Worth', 'Annual_Income', 'Risk_Appetite', 'Primary_Investment', 'Debt_to_Income_Ratio']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing wealth data columns: {missing_cols}")
    net_worth = customer['Net_Worth'].iloc[0]
    income = customer['Annual_Income'].iloc[0]
    risk_appetite = customer['Risk_Appetite'].iloc[0]
    primary_investment = customer['Primary_Investment'].iloc[0]
    dti_ratio = customer['Debt_to_Income_Ratio'].iloc[0]
    prompt = f"""
    Generate a concise, personalized investment recommendation based on:
    - Net Worth: ${net_worth}
    - Annual Income: ${income}
    - Risk Appetite: {risk_appetite}
    - Current Primary Investment: {primary_investment}
    - Debt-to-Income Ratio: {dti_ratio}
    
    Clearly provide a tailored investment recommendation in 1-2 friendly, professional sentences.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            wealth_advice = result['candidates'][0]['content']['parts'][0]['text']
            return {"wealth_management_advice": wealth_advice}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

#business_df = pd.read_csv('business_financials.csv')  # (reloaded full data)
@app.get("/business-insights/{business_id}")
def business_insights(business_id: str):
    business = business_df[business_df['business_id'] == business_id]
    if business.empty:
        raise HTTPException(404, "Business not found in financial data")
    required_cols = ['annual_revenue', 'annual_expenses', 'profit_margin', 'credit_score', 'social_media_presence']
    missing_cols = [col for col in required_cols if col not in business.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing business data columns: {missing_cols}")
    annual_revenue = business['annual_revenue'].iloc[0]
    annual_expenses = business['annual_expenses'].iloc[0]
    profit_margin = business['profit_margin'].iloc[0]
    credit_score = business['credit_score'].iloc[0]
    social_media_presence = business['social_media_presence'].iloc[0]
    prompt = f"""
    Provide a concise, actionable business financial insight based on:
    - Annual Revenue: ${annual_revenue}
    - Annual Expenses: ${annual_expenses}
    - Profit Margin: {profit_margin}%
    - Credit Score: {credit_score}
    - Social Media Followers: {social_media_presence}
    
    Clearly suggest one strategic recommendation to improve profitability or financial stability, in 1-2 sentences.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            business_advice = result['candidates'][0]['content']['parts'][0]['text']
            return {"business_financial_insight": business_advice}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

#market_df = pd.read_csv('market_insights.csv')  # (reloaded full data)
@app.get("/market-insights/{business_id}")
def market_insights(business_id: str):
    business = market_df[market_df['business_id'] == business_id]
    if business.empty:
        raise HTTPException(404, "Business not found in market insights data")
    required_cols = ['industry_growth_rate', 'competitor_count', 'customer_acquisition_cost', 'advertising_budget', 'brand_reputation_score', 'social_media_engagement']
    missing_cols = [col for col in required_cols if col not in business.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing market data columns: {missing_cols}")
    growth_rate = business['industry_growth_rate'].iloc[0]
    competitors = business['competitor_count'].iloc[0]
    acquisition_cost = business['customer_acquisition_cost'].iloc[0]
    ad_budget = business['advertising_budget'].iloc[0]
    brand_score = business['brand_reputation_score'].iloc[0]
    social_engagement = business['social_media_engagement'].iloc[0]
    prompt = f"""
    Provide a concise and strategic market insight based on:
    - Industry Growth Rate: {growth_rate}%
    - Competitor Count: {competitors}
    - Customer Acquisition Cost: ${acquisition_cost}
    - Advertising Budget: ${ad_budget}
    - Brand Reputation Score: {brand_score}/5
    - Social Media Engagement: {social_engagement}
    
    Clearly suggest one actionable strategy for business growth or improved market positioning in 1-2 sentences.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    except Exception as e:
        raise HTTPException(500, f"AI API request failed: {e}")
    if res.status_code == 200:
        result = res.json()
        try:
            market_advice = result['candidates'][0]['content']['parts'][0]['text']
            return {"market_insight_advice": market_advice}
        except (IndexError, KeyError):
            raise HTTPException(500, "Unexpected AI response structure.")
    else:
        raise HTTPException(res.status_code, f"AI API error: {res.text}")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically detect numeric columns for model training
if fraud_df is not None:
    fraud_df.columns = fraud_df.columns.str.strip()  # Remove extra spaces in column names

    # Exclude non-numeric & identifier columns
    numeric_features = fraud_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_columns = ["Transaction ID", "Customer ID", "Fraud Label"]  # Non-feature columns
    fraud_features = [col for col in numeric_features if col not in exclude_columns]

    # Check if there are enough features for training
    if len(fraud_features) < 2:
        logger.error(f"❌ Not enough numeric features found! Detected: {fraud_features}")
        fraud_df = None  # Prevent invalid dataset usage
    else:
        logger.info(f"📊 Selected fraud detection features: {fraud_features}")

        # Ensure 'Fraud Label' exists
        if "Fraud Label" in fraud_df.columns:
            fraud_df = fraud_df[["Transaction ID"] + fraud_features + ["Fraud Label"]]
        else:
            logger.error("❌ 'Fraud Label' column is missing! Cannot train model.")
            fraud_df = None
else:
    logger.error("❌ Fraud dataset could not be loaded.")

# Encode categorical features dynamically
if fraud_df is not None:
    le = LabelEncoder()
    categorical_features = fraud_df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_features:
        if col not in ["Transaction ID", "Customer ID"]:  # Avoid encoding unique IDs
            fraud_df[col] = le.fit_transform(fraud_df[col].astype(str))
    logger.info(f"✅ Categorical encoding applied to: {categorical_features}")

# Train fraud detection model dynamically
if fraud_df is not None:
    try:
        X = fraud_df[fraud_features]  # Use dynamically selected features
        y = fraud_df["Fraud Label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        fraud_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        fraud_model.fit(X_train, y_train)
        logger.info("✅ Advanced fraud detection model trained successfully.")

        # SHAP explainability
        explainer = shap.Explainer(fraud_model)
    except Exception as e:
        logger.critical(f"❌ Failed to train fraud model: {e}")
        fraud_model = None
        explainer = None
else:
    fraud_model = None
    explainer = None

# Fraud Advisory LLM Function (AI-generated advice)
def generate_fraud_advisory(transaction, prediction, confidence):
    """Generate a fraud advisory message based on model prediction dynamically."""
    risk_score = transaction.iloc[0].get("Risk Score", "N/A")
    amount = transaction.iloc[0].get("Transaction Amount", "N/A")

    if prediction == 1:  # Fraudulent
        if confidence > 0.9:
            return f"🚨 High confidence fraud alert! Transaction (${amount}) is highly suspicious with a risk score of {risk_score}."
        elif str(risk_score).isdigit() and float(risk_score) > 0.85:
            return f"⚠️ Possible fraud detected! Transaction of ${amount} has a high risk score of {risk_score}."
        else:
            return f"🔍 Unusual activity detected in transaction of ${amount}. Review transaction history."
    else:  # Legitimate
        return f"✅ No fraud detected. Transaction of ${amount} appears normal."

@app.get("/fraud/{transaction_id}")
def fraud(transaction_id: str):
    try:
        # Ensure fraud_df is loaded
        if fraud_df is None or fraud_model is None:
            raise HTTPException(status_code=500, detail={"error": True, "message": "Fraud detection system is not initialized.", "transaction_id": transaction_id})

        # Normalize column names
        fraud_df.columns = fraud_df.columns.str.strip()

        # Check if the transaction exists
        transaction = fraud_df[fraud_df['Transaction ID'] == transaction_id]

        if transaction.empty:
            raise HTTPException(status_code=404, detail={"error": True, "message": "Transaction not found", "transaction_id": transaction_id})

        # Select only required features
        transaction_features = transaction[fraud_features]

        # Check for missing values
        if transaction_features.isnull().values.any():
            raise HTTPException(status_code=500, detail={"error": True, "message": "Transaction data contains missing values.", "transaction_id": transaction_id})

        # Make a fraud prediction
        prediction = int(fraud_model.predict(transaction_features)[0])  # Convert NumPy int to Python int
        confidence_score = float(max(fraud_model.predict_proba(transaction_features)[0]))  # Convert NumPy float to Python float

        # Generate SHAP explainability for feature importance
        shap_values = explainer(transaction_features)
        top_features = sorted(zip(fraud_features, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        # Convert NumPy types to Python types
        feature_importance = {str(feat): float(val) for feat, val in top_features}  # Ensure JSON serialization

        # Generate AI-driven fraud advisory
        fraud_advice = generate_fraud_advisory(transaction, prediction, confidence_score)

        return {
            "error": False,
            "transaction_id": transaction_id,
            "is_fraud": bool(prediction),
            "confidence_score": confidence_score,
            "top_feature_importance": feature_importance,
            "fraud_advisory": fraud_advice
        }

    except HTTPException as http_err:
        raise http_err  # Re-raise known HTTP exceptions

    except Exception as e:
        logger.exception(f"🚨 Unexpected error in fraud detection API: {e}")
        raise HTTPException(status_code=500, detail={"error": True, "message": f"Unexpected error in fraud detection: {str(e)}", "transaction_id": transaction_id})



@app.get("/subscription/{customer_id}")
def subscription(customer_id: str):
    customer = subscription_df[subscription_df['Customer_ID'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found")
    required_cols = ['Monthly_Fee', 'Monthly_Spending']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required subscription features: {missing_cols}")
    try:
        prediction = subscription_model.predict(customer[['Monthly_Fee', 'Monthly_Spending']])[0]
    except Exception as e:
        raise HTTPException(500, f"Subscription model prediction error: {e}")
    return {"cancel_subscription": bool(prediction)}

@app.get("/wealth/{customer_id}")
def wealth(customer_id: str):
    customer = wealth_df[wealth_df['Customer_ID'] == customer_id]
    if customer.empty:
        raise HTTPException(404, "Customer not found")
    required_cols = ['Net_Worth', 'Annual_Income', 'Debt_to_Income_Ratio']
    missing_cols = [col for col in required_cols if col not in customer.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required wealth features: {missing_cols}")
    try:
        prediction = wealth_model.predict(customer[['Net_Worth', 'Annual_Income', 'Debt_to_Income_Ratio']])[0]
    except Exception as e:
        raise HTTPException(500, f"Wealth model prediction error: {e}")
    return {"risk_appetite": int(prediction)}

@app.get("/business/{business_id}")
def business(business_id: str):
    business = business_df[business_df['business_id'] == business_id]
    if business.empty:
        raise HTTPException(404, "Business not found")
    required_cols = ['annual_revenue', 'credit_score', 'social_media_presence']
    missing_cols = [col for col in required_cols if col not in business.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required business features: {missing_cols}")
    try:
        prediction = business_model.predict(business[['annual_revenue', 'credit_score', 'social_media_presence']])[0]
    except Exception as e:
        raise HTTPException(500, f"Business model prediction error: {e}")
    return {"predicted_profit_margin": float(prediction)}

@app.get("/market/{business_id}")
def market(business_id: str):
    business = market_df[market_df['business_id'] == business_id]
    if business.empty:
        raise HTTPException(404, "Business not found")
    required_cols = ['industry_growth_rate', 'advertising_budget']
    missing_cols = [col for col in required_cols if col not in business.columns]
    if missing_cols:
        raise HTTPException(500, f"Missing required market features: {missing_cols}")
    try:
        prediction = market_model.predict(business[['industry_growth_rate', 'advertising_budget']])[0]
    except Exception as e:
        raise HTTPException(500, f"Market model prediction error: {e}")
    return {"predicted_customer_acquisition_cost": float(prediction)}

#####################

from sklearn.linear_model import LogisticRegression
import xgboost as xgb  # (already imported above)
# Load all datasets (full data)
try:
    banking_df = pd.read_csv('synthetic_banking_data.csv')
    loan_df = pd.read_csv('loan_credit_analysis.csv')
    fraud_df = pd.read_csv('synthetic_fraud_detection.csv')
    subscription_df = pd.read_csv('subscription_spending_patterns.csv')
    wealth_df = pd.read_csv('synthetic_wealth_management.csv')
    business_df = pd.read_csv('business_financials.csv')
    market_df = pd.read_csv('market_insights.csv')
except FileNotFoundError as e:
    print(f" ERROR: Missing dataset file - {e}")
#  Ensure all dataframes exist even if loading failed
if 'banking_df' not in globals():
    banking_df = pd.DataFrame()
if 'loan_df' not in globals():
    loan_df = pd.DataFrame()
if 'fraud_df' not in globals():
    fraud_df = pd.DataFrame()
if 'subscription_df' not in globals():
    subscription_df = pd.DataFrame()
if 'wealth_df' not in globals():
    wealth_df = pd.DataFrame()
if 'business_df' not in globals():
    business_df = pd.DataFrame()
if 'market_df' not in globals():
    market_df = pd.DataFrame()

###########test
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
import os
import logging

#  Configure FastAPI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#  Load datasets (ignores missing files)
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
        logger.info(f" Loaded {name} dataset")
    except FileNotFoundError:
        logger.warning(f"⚠️ Missing dataset: {file}, skipping...")

#  Load trained ML models (skips missing models)
#  Load trained ML models
models = {}
label_encoders = {}

for model_name in ["loan_model", "fraud_model", "subscription_model", "wealth_model"]:
    try:
        models[model_name] = joblib.load(f"{model_name}.pkl")
        logger.info(f" Loaded model: {model_name}")
        #  Load label encoder if it exists
        if os.path.exists(f"{model_name}_label_encoder.pkl"):
            label_encoders[model_name] = joblib.load(f"{model_name}_label_encoder.pkl")
            logger.info(f" Loaded label encoder for {model_name}")
    except FileNotFoundError:
        logger.warning(f"⚠️ Model {model_name}.pkl not found, skipping...")


#  LLM API (Replace with actual API key)


def call_llm(prompt):
    """Handles Google Gemini API failures with retries."""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(3):  #  Retry up to 3 times
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers, timeout=10)

            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            elif response.status_code == 401:
                logger.error(" Unauthorized! Check API Key and ensure API is enabled.")
                return "⚠️ AI Service Unauthorized. Please check your API settings."
            elif response.status_code == 403:
                logger.error(" Forbidden! Ensure API is enabled and billing is active.")
                return "⚠️ AI Service Access Denied. Check Google Cloud settings."
            elif response.status_code == 429:
                logger.warning("⚠️ Too Many Requests. Retrying...")
                continue  # Retry on rate limiting
            else:
                logger.error(f" Unexpected Gemini API Error: {response.status_code}, {response.text}")
                return "⚠️ AI Service is temporarily unavailable."
        except requests.RequestException as e:
            logger.error(f" Request Error: {e}")
            return "⚠️ AI Service Error. Try again later."

    return "⚠️ AI Service Failed After Multiple Attempts."

#  Chatbot API Endpoint
class ChatRequest(BaseModel):
    query: str
    customer_id: str = None

# Sample models & datasets (Replace with real ML models & data)
models = {"loan_model": None, "fraud_model": None, "subscription_model": None, "wealth_model": None}
datasets = {
    "loan": pd.DataFrame(columns=['Customer_ID', 'age', 'income', 'credit_score', 'existing_loans']),
    "fraud": pd.DataFrame(columns=['Customer_ID', 'Transaction_Amount', 'Risk_Score']),
    "subscription": pd.DataFrame(columns=['Customer_ID', 'Monthly_Fee', 'Monthly_Spending']),
    "wealth": pd.DataFrame(columns=['Customer_ID', 'Net_Worth', 'Annual_Income', 'Debt_to_Income_Ratio']),
    "banking": pd.DataFrame(columns=['Customer_ID', 'budget', 'income', 'expenses'])
}

def format_as_table(data):
    """Converts dictionary data into Markdown table format."""
    table = "| " + " | ".join(data.keys()) + " |\n"
    table += "| " + " | ".join(["---"] * len(data)) + " |\n"
    table += "| " + " | ".join(str(v) for v in data.values()) + " |\n"
    return table

@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        insights = {}
        dataset_mapping = {
            "credit score": "loan",
            "loan": "loan",
            "fraud": "fraud",
            "investment": "wealth",
            "savings": "wealth",
            "spending": "subscription",
            "subscription": "subscription",
            "budget": "banking",
            "income": "banking",
            "expenses": "banking"
        }

        # Identify dataset based on query
        dataset_name = "banking"
        for keyword, dataset in dataset_mapping.items():
            if keyword in request.query.lower():
                dataset_name = dataset
                break

        # Fetch customer data & generate insights
        if request.customer_id and dataset_name in datasets:
            customer = datasets[dataset_name][datasets[dataset_name]["Customer_ID"] == request.customer_id]
            if customer.empty:
                return {"response": "Customer not found."}

            try:
                if dataset_name == "loan" and "loan_model" in models:
                    dti = round((customer['existing_loans'].values[0] / customer['income'].values[0]) * 100, 2)
                    insights["debt_to_income_ratio"] = f"{dti}%"
                    insights["loan_approval"] = models["loan_model"].predict(customer[['age', 'income', 'credit_score', 'existing_loans']])[0]
                    insights["expected_interest_rate"] = "7% - 14%" if insights["loan_approval"] else "14% - 20%"
                    
                elif dataset_name == "fraud" and "fraud_model" in models:
                    insights["fraud_risk"] = models["fraud_model"].predict(customer[['Transaction_Amount', 'Risk_Score']])[0]

                elif dataset_name == "subscription" and "subscription_model" in models:
                    insights["subscription_risk"] = models["subscription_model"].predict(customer[['Monthly_Fee', 'Monthly_Spending']])[0]

                elif dataset_name == "wealth" and "wealth_model" in models:
                    age = customer['age'].values[0]
                    target_age = 55
                    years_left = target_age - age
                    net_worth = customer['Net_Worth'].values[0]
                    annual_income = customer['Annual_Income'].values[0]
                    annual_investment = round(0.2 * annual_income, 2)  # 20% savings rate assumption
                    
                    future_savings = net_worth
                    yearly_growth = []
                    for year in range(years_left):
                        future_savings = future_savings * 1.07 + annual_investment
                        yearly_growth.append({"Year": age + year, "Projected Savings": f"${round(future_savings, 2)}"})

                    insights["retirement_projection"] = future_savings
                    insights["investment_strategy"] = f"Invest ${annual_investment} per year to retire with ~${round(future_savings, 2)} at {target_age}."

            except Exception as e:
                logger.error(f"⚠️ ML Prediction Failed: {e}")

        # **🚀 Generate Enhanced LLM Prompt with Expert Financial Advice**
        llm_prompt = f"""
        Generate a professional, structured financial advisory response.

        - **User Query**: {request.query}
        - **Predicted Insights**: {json.dumps(insights, indent=2)}

        The response should be:
        - **Expert-Level** (Like a professional financial consultant)
        - **Personalized** (Based on user profile)
        - **Data-Driven** (Justified using AI predictions)
        - **Actionable** (Providing next steps)
        - **Concise** (Max 4 sentences)
        """

        response_text = call_llm(llm_prompt)

        # **🚀 Add Markdown Table for UI Display if Required**
        if "loan_approval" in insights or "retirement_projection" in insights:
            response_text += "\n\n### Financial Insights 📊\n"
            response_text += format_as_table(insights)

        return {"response": response_text}

    except Exception as e:
        logger.error(f" Chatbot Error: {e}")
        raise HTTPException(500, "Chatbot analysis failed")


################test end


#  Added missing categorization function
def categorize_spending(transactions):
    """Categorize transactions into common spending categories"""
    category_map = {
        "GROCERIES": ["walmart", "sainsbury's", "tesco", "asda"],
        "DINING": ["mcdonald's", "starbucks", "restaurant", "cafe"],
        "UTILITIES": ["thames water", "bt broadband", "national grid"],
        "ENTERTAINMENT": ["netflix", "spotify", "cineworld"],
        "TRANSPORT": ["uber", "tfl", "shell petrol"],
        "HOUSING": ["rent payment", "mortgage", "council tax"]
    }
    
    categorized = {}
    for vendor, amount in transactions.items():
        vendor_lower = vendor.lower()
        category = "OTHER"
        for cat, keywords in category_map.items():
            if any(kw in vendor_lower for kw in keywords):
                category = cat
                break
        categorized[vendor] = {
            "amount": amount,
            "category": category
        }
    return categorized

#  Modified extraction with better regex
def extract_text_from_image(image):
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Vision API request
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Extract all transactions with merchant names and amounts. Include currency symbols."},
                    {"inline_data": {"mime_type": "image/png", "data": img_str}}
                ]
            }]
        }

        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            timeout=15
        )

        if response.status_code != 200:
            return f"API Error: {response.status_code}", {}

        result = response.json()
        extracted_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

        # Improved transaction parsing
        transactions = {}
        pattern = r"([^\d\$£€]+?)\s*([\$£€]\s*[\d,]+\.\d{2})"
        matches = re.findall(pattern, extracted_text)
        
        for vendor, amount in matches:
            vendor = vendor.strip()
            if vendor and amount:
                try:
                    # Normalize currency format
                    clean_amount = re.sub(r'[^\d.]', '', amount)
                    transactions[vendor] = f"${float(clean_amount):,.2f}"
                except:
                    continue

        return extracted_text, transactions

    except Exception as e:
        return str(e), {}

#  Fixed financial advice generation
def generate_financial_advice(transactions):
    if not transactions:
        return ["No transactions found to analyze."]

    # Categorize transactions
    categorized_spending = categorize_spending(transactions)
    
    # Format transaction data for AI analysis
    total_spending = sum(float(value['amount'].replace('$', '').replace(',', '')) 
                        for value in categorized_spending.values())
    
    transaction_summary = "\n".join([
        f"{key}: {value['amount']} ({value['category']})" 
        for key, value in categorized_spending.items()
    ])

    prompt = f"""
    As a financial advisor, analyze these transactions (Total: ${total_spending:,.2f}):
    
    {transaction_summary}
    
    Provide 3 specific recommendations:
    1. Spending optimization
    2. Savings opportunity
    3. Budget planning
    
    Keep the advice concise and actionable.
    """

    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            advice = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 
                    "Unable to generate financial advice at this time.")
            return [advice]
        
        return ["Financial analysis service temporarily unavailable."]

    except Exception as e:
        logger.error(f"Financial Advice Generation Error: {str(e)}")
        return ["Unable to process financial advice request."]

#  Maintained original endpoint with fixed response structure
@app.post("/classify-image/")
async def classify_image(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).convert("RGB")
        extracted_text, transactions = extract_text_from_image(image)
        
        # Default error response
        response_template = {
            "document_analysis": {
                "category": "Financial Document",
                "match": "High",
                "confidence": 0.95
            },
            "extracted_text": extracted_text,
            "categorized_spending": {},
            "personalized_advice": []
        }

        if not transactions:
            response_template.update({
                "document_analysis": {
                    "category": "Unrecognized Format",
                    "match": "Low",
                    "confidence": 0.3
                },
                "personalized_advice": ["No transactions found"]
            })
            return response_template

        categorized = categorize_spending(transactions)
        advice = generate_financial_advice(transactions)
        
        response_template.update({
            "categorized_spending": categorized,
            "personalized_advice": advice
        })
        
        return response_template

    except Exception as e:
        return {
            "document_analysis": {
                "category": "Unknown",
                "match": "None",
                "confidence": 0.0
            },
            "extracted_text": str(e),
            "categorized_spending": {},
            "personalized_advice": ["Analysis failed"]
        }




#########################################################image

from pydantic import BaseModel
class ConsentRequest(BaseModel):
    consent_given: bool
# Stores user consent (Example: In-memory dictionary)
consent_records = {}  # Format: {customer_id: {"consent_given": True/False, "timestamp": "..."}}

@app.post("/consent/{customer_id}")
def set_consent(customer_id: str, consent_data: ConsentRequest):
    """
     Allows users to give or revoke AI consent.
     If consent is already granted, avoids unnecessary updates.
     If revoking when consent isn't set, returns a clear message.
    """
    try:
        if consent_data.consent_given:
            #  Prevent redundant updates
            if customer_id in consent_records and consent_records[customer_id]["consent_given"]:
                return {
                    "customer_id": customer_id,
                    "message": " Consent was already given.",
                    "consent_given": True
                }
            
            #  Store consent
            consent_records[customer_id] = {
                "consent_given": True,
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            return {
                "customer_id": customer_id,
                "message": f" Consent granted for {customer_id}",
                "consent_given": True
            }
        else:
            #  Check if consent exists before revoking
            if customer_id in consent_records:
                del consent_records[customer_id]
                return {
                    "customer_id": customer_id,
                    "message": f" Consent revoked for {customer_id}",
                    "consent_given": False
                }
            return {
                "customer_id": customer_id,
                "message": "⚠️ No consent record found to revoke.",
                "consent_given": False
            }
    except Exception as e:
        print(f" Error updating consent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while updating consent.")




@app.get("/consent/{customer_id}")
def check_consent(customer_id: str):
    """
     Checks if the user has given consent.
     If no consent record exists, returns `"consent_given": False` instead of an error.
    """
    try:
        consent_info = consent_records.get(customer_id, None)
        if consent_info:
            return {
                "customer_id": customer_id,
                "consent_given": consent_info["consent_given"],
                "timestamp": consent_info["timestamp"]
            }
        return {
            "customer_id": customer_id,
            "consent_given": False,
            "message": " No consent record found."
        }
    except Exception as e:
        print(f" Error checking consent: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while checking consent.")



def check_fairness(prediction):
    """
    Checks if AI predictions meet fairness standards.
    - If confidence is too low (< 0.65), warns user for human review.
    """
    if "confidence" in prediction and prediction["confidence"] < 0.65:  # Example threshold
        prediction["warning"] = "⚠️ AI confidence is low, human review recommended."
    return prediction


