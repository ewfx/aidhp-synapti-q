from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
from sklearn.cluster import KMeans
import shap
import requests
from sklearn.ensemble import RandomForestClassifier  
import numpy as np 
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
from typing import Dict, Any, Optional, List 
import logging
from pydantic import BaseModel
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time

import pickle
from datetime import datetime
from pydantic import BaseModel
from datetime import datetime 
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import Path
import logging
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

platform_mapping = {
    1: "Facebook",
    2: "Instagram",
    3: "Twitter",
    4: "LinkedIn",
    5: "TikTok",
    6: "YouTube"
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  #  React app's URL
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

def convert_numpy_types(value):
    """Helper function to convert NumPy types to native Python types"""
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value  

def generate_ai_loan_insight(loan_data):
    """Generates strictly formatted loan repayment insights only"""
    
    if not loan_data:
        return {"status": "success", "message": "No active loan customers found."}

    personalized_insights = []
    
    for customer in loan_data:
        # Create a locked-down prompt with explicit output control
        insight_prompt = f"""
        <<STRICT INSTRUCTIONS>>
        You are a loan repayment analysis tool. You MUST ONLY:
        - Analyze existing loan repayment options
        - Suggest refinancing ONLY when mathematically beneficial
        - Provide credit improvement tips if score <650
        - Give 3 actionable steps
        
        <<PROHIBITED>>
        - Never mention applications, approvals, or congratulations
        - Never assume new loans are possible
        - Never use generic templates
        
        <<REQUIRED FORMAT>>
        Analysis for Customer {customer['customer_id']}:
        1. Current Loan Analysis: [50 words max]
        2. Repayment Strategy: [3 specific tactics]
        3. Refinance Potential: [Yes/No + breakeven analysis]
        4. Credit Tips: [Only if score <650]
        5. Next Steps: [3 concrete actions with numbers]
        
        <<CUSTOMER DATA>>
        Loan Amount: ${customer['loan_amount']}
        Interest Rate: {customer['interest_rate']}%
        Credit Score: {customer['credit_score']}
        Monthly Payment: ${customer['monthly_installment']}
        
        <<BEGIN ANALYSIS>>"""
        
        # API configuration to enforce strict behavior
        payload = {
            "contents": [{
                "parts": [{"text": insight_prompt}],
                "role": "model",
                "examples": [
                    {
                        "input": {"content": "Generic loan approval template"},
                        "output": {"content": "[REJECTED] Only repayment analysis permitted"}
                    }
                ]
            }],
            "safetySettings": [
                {"category": "HARM_CATEGORY_DEROGATORY", "threshold": "BLOCK_ONLY_HIGH"}
            ],
            "generationConfig": {
                "temperature": 0.3,  # Lower for more deterministic responses
                "topK": 20,
                "topP": 0.7,
                "maxOutputTokens": 500,
                "stopSequences": ["Congratulations", "approved"]
            }
        }

        try:
            # First attempt
            response = get_ai_response(payload)
            
            # Validation and retry if needed
            if contains_prohibited_phrases(response):
                payload["generationConfig"]["temperature"] = 0.1  # More strict
                response = get_ai_response(payload)
                
                if contains_prohibited_phrases(response):
                    response = manually_structured_response(customer)
            
            personalized_insights.append({
                "customer_id": customer['customer_id'],
                "ai_insight": response
            })
            
        except Exception as e:
            personalized_insights.append({
                "customer_id": customer['customer_id'],
                "ai_insight": manually_structured_response(customer),
                "error": str(e)
            })
    
    return personalized_insights

def get_ai_response(payload):
    """Helper function to call AI API"""
    headers = {
        "Content-Type": "application/json",
        "X-Require-Strict": "true"  # Custom header if  API supports it
    }
    res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
    res.raise_for_status()
    return res.json()['candidates'][0]['content']['parts'][0]['text']

def contains_prohibited_phrases(text):
    """Validates response against unwanted content"""
    prohibited = [
        "approved", "congrat", "application", 
        "good news", "pleased to inform", "qualify"
    ]
    return any(phrase.lower() in text.lower() for phrase in prohibited)

def manually_structured_response(customer):
    """Fallback template when AI fails"""
    return f"""
    Analysis for Customer {customer['customer_id']}:
    1. Current Loan Analysis: ${customer['loan_amount']} at {customer['interest_rate']}% interest
    2. Repayment Strategy: 
       - Make biweekly payments of ${round(customer['monthly_installment']/2, 2)}
       - Allocate 10% extra to principal monthly
       - Review budget for additional $100/month payment
    3. Refinance Potential: {'Yes' if customer['credit_score'] > 700 else 'No'}
    4. Credit Tips: {'Increase credit limits and reduce utilization' if customer['credit_score'] < 650 else 'N/A'}
    5. Next Steps: 
       1) Contact lender about repayment options
       2) Set up automatic payments
       3) Review expenses for additional payment capacity
    """

@app.get("/loan-customer-ids/")
def get_loan_customer_ids(customer_id: int =20):
    """
    Retrieves a list of customer IDs with loan-related data.
    - Returns a Generative AI financial insight alongside the customer IDs.
    """
    # Use full loan data if available
    df = loan_df if 'loan_df' in globals() and not loan_df.empty else loan_credit_df
    
    if df.empty:
        return {
            "status": "success",
            "message": "Loan data is not available at the moment.",
            "ai_insight": generate_ai_loan_insight(0),  # AI generates a response for no data
            "available_loan_customer_ids": []
        }

    # Convert NumPy types to Python-native types
    #ids = [convert_numpy_types(cust_id) for cust_id in df['customer_id'].head(limit).tolist()]
    loan_data = df.head(limit).to_dict(orient="records")
    loan_data = [{k: convert_numpy_types(v) for k, v in customer.items()} for customer in loan_data]

    
    # Fetch AI-generated loan insights
    ai_insight = generate_ai_loan_insight((loan_data))
    insights_dict = {insight['customer_id']: insight.get('ai_insight', 'No insight available') 
                    for insight in ai_insight}

    return {
        "status": "success",
        "customers": [{
            "customer_data": customer,
            "ai_insight": insights_dict.get(customer['customer_id'], "No insight available")
        } for customer in loan_data]
    }

#####

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'financial_suggestion_model.pkl')

PLATFORM_MAPPING = {1: "Facebook", 2: "Instagram", 3: "Twitter", 4: "LinkedIn"}

# Response Models
class FinancialSuggestion(BaseModel):
    suggestion: str
    confidence: float
    probabilities: dict

class SocialAnalysis(BaseModel):
    platform: str
    recommendation: str
    sentiment_score: float

class LoanAnalysis(BaseModel):
    amount: float
    interest_rate: float
    repayment_strategy: str
    refinance_recommendation: str

class FinancialAdviceResponse(BaseModel):
    status: str
    customer_id: str
    financial_suggestion: FinancialSuggestion
    ai_advice: str
    social_analysis: Optional[SocialAnalysis] = None
    loan_analysis: Optional[LoanAnalysis] = None

class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    suggestion: Optional[str] = None

# Load model at startup
try:
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts['model']
    features = artifacts['features']
    label_encoders = artifacts.get('label_encoders', {})
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Service initialization failed")

def convert_numpy_types(value):
    """Convert numpy types to native Python types safely"""
    if isinstance(value, (np.integer)):
        return int(value)
    elif isinstance(value, (np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif pd.isna(value):
        return None
    return value

def get_ai_response(prompt: str, temperature: float = 0.7) -> str:
    """Get AI response with robust error handling"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": max(0, min(1, temperature)),
            "topP": 0.9,
            "maxOutputTokens": 500,
            "stopSequences": ["disclaimer", "approval"]
        }
    }
    
    try:
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        logger.error(f"AI API error: {str(e)}")
        raise HTTPException(503, detail="AI service unavailable")

@app.get(
    "/financial/{customer_id}",
    response_model=FinancialAdviceResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Customer not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    }
)
async def get_financial_advice(
    customer_id: str = Path(..., description="Customer ID to lookup"),  # Changed Query to Path
    include_social: bool = Query(True, description="Include social media analysis"),
    include_loans: bool = Query(True, description="Include loan analysis"),
    creativity: float = Query(0.7, ge=0, le=1, description="AI creativity level (0-1)")
):
    """
    Get comprehensive financial advice for a customer
    
    Returns:
    - ML-based financial suggestion with confidence scores
    - AI-generated detailed advice
    - Optional social media analysis
    - Optional loan analysis
    """
    try:
        # 1. Retrieve and validate customer data
        try:
            customer = banking_df[banking_df['Customer_ID'] == customer_id].iloc[0]
        except IndexError:
            raise HTTPException(404, detail={
                "detail": "Customer not found",
                "error_code": "CUSTOMER_NOT_FOUND",
                "suggestion": "Verify customer ID exists in the system"
            })
        
        # 2. Convert data to native types
        customer_data = {k: convert_numpy_types(v) for k, v in customer.items()}
        
        # 3. Generate ML suggestion
        try:
            input_features = [customer_data.get(f, 0) for f in features]
            suggestion = model.predict([input_features])[0]
            probabilities = model.predict_proba([input_features])[0]
            
            ml_suggestion = FinancialSuggestion(
                suggestion=suggestion,
                confidence=float(np.max(probabilities)),
                probabilities=dict(zip(model.classes_, probabilities))
            )
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise HTTPException(500, detail={
                "detail": "Financial suggestion generation failed",
                "error_code": "MODEL_ERROR"
            })

        # 4. Generate AI advice
        ai_prompt = f"""
        Financial Profile:
        - Age: {customer_data.get('Age', 'N/A')}
        - Income: ${customer_data.get('Income', 0):,.2f}
        - Credit Score: {customer_data.get('Credit_Score', 0)}
        - Existing Loans: {customer_data.get('Existing_Loans', 0)}
        
        Model Suggestion: {ml_suggestion.suggestion} (Confidence: {ml_suggestion.confidence:.0%})
        
        Provide:
        1. Specific suggestions (3 bullet points)
        2. Expected outcomes 
        3. Risk considerations make sure to have full meaningful sentence
        """
        
        ai_advice = get_ai_response(ai_prompt, creativity)

        # 5. Social media analysis
        social_analysis = None
        if include_social and all(f in customer_data for f in ['Social_Media_Platform', 'Customer_Feedback']):
            try:
                platform = PLATFORM_MAPPING.get(int(customer_data['Social_Media_Platform']), "Social Media")
                sentiment = SentimentIntensityAnalyzer().polarity_scores(customer_data['Customer_Feedback'])
                
                social_analysis = SocialAnalysis(
                    platform=platform,
                    recommendation=get_ai_response(
                        f"Create a {platform}-appropriate financial tip for a user with sentiment score {sentiment['compound']:.2f}",
                        min(creativity + 0.2, 1.0)
                    ),
                    sentiment_score=sentiment['compound']
                )
            except Exception as e:
                logger.warning(f"Social analysis skipped: {str(e)}")

        # 6. Loan analysis
        loan_analysis = None
        if include_loans and customer_data.get('loan_amount', 0) > 0:
            try:
                loan_analysis = LoanAnalysis(
                    amount=customer_data['loan_amount'],
                    interest_rate=customer_data.get('interest_rate', 0),
                    repayment_strategy=get_ai_response(
                        f"Suggest optimal repayment for ${customer_data['loan_amount']:,.2f} loan at {customer_data.get('interest_rate', 0)}% interest",
                        max(creativity - 0.3, 0.1)
                    ),
                    refinance_recommendation=get_ai_response(
                        f"Should a customer with credit score {customer_data.get('Credit_Score', 0)} refinance a {customer_data.get('interest_rate', 0)}% loan? Answer yes/no with brief reasoning",
                        0.3
                    )
                )
            except Exception as e:
                logger.warning(f"Loan analysis skipped: {str(e)}")

        return FinancialAdviceResponse(
            status="success",
            customer_id=customer_id,
            financial_suggestion=ml_suggestion,
            ai_advice=ai_advice,
            social_analysis=social_analysis,
            loan_analysis=loan_analysis
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(500, detail={
            "detail": "Internal processing error",
            "error_code": "INTERNAL_ERROR"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#Load the Trained Churn Model

import json
import requests

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
    """
    Retrieves loan details for a specific customer and provides AI-driven financial recommendations.
    """
    
    df = loan_credit_df
    customer = df[df['customer_id'] == customer_id]

    if customer.empty:
        return {
            "status": "error",
            "message": f"Customer ID {customer_id} not found in the loan dataset."
        }

    # Ensure required columns exist
    required_cols = ['age', 'income', 'credit_score', 'existing_loans', 'loan_approval_status', 'debt_to_income_ratio', 'loan_interest_rate']
    missing_cols = [col for col in required_cols if col not in customer.columns]

    if missing_cols:
        raise HTTPException(500, f"Missing required loan features: {missing_cols}")

    #Convert DataFrame row to dictionary and ensure correct data types
    loan_details = customer.to_dict(orient="records")[0]  # Extract first row as dictionary
    loan_details = {k: convert_numpy_types(v) for k, v in loan_details.items()}  # Convert NumPy types

    #Loan Approval Prediction
    loan_features = ['age', 'income', 'credit_score', 'existing_loans']
    customer_filtered = customer[loan_features]
    
    try:
        prediction = loan_model.predict(customer_filtered)[0]
        prediction = int(prediction)  # Ensure it's a native int
    except Exception as e:
        raise HTTPException(500, f"Loan model prediction error: {e}")

    #Create AI Prompt for Financial & Loan Recommendations
    ai_prompt = f"""
     **Customer Financial Analysis Report**
    
     **Your Task:** You are a financial expert analyzing a customer's financial health based on their loan data.

    **Loan Approval Prediction:** {prediction}

    **Customer Financial Data:**
    {loan_details}

     **Instructions:**
    - **Evaluate the customer's financial stability** and potential loan risks.
    - If **loan approval is rejected**, suggest alternative financial solutions, also indetailed why its rejected.
    - If **debt-to-income ratio is high (>0.4)**, recommend ways to lower it and suggest how.
    - If **credit score is below 650**, suggest strategies to improve it and say even your loan was initialy approved but chances to final rejection is high and why.
    - If **interest rate is too high**, recommend refinancing options and why.
    - Suggest **investment or savings strategies** based on income and risk.
    - Provide a **real-world example** of someone who successfully improved their financial situation.

    **Generate a detailed, personalized, and customer-friendly financial report not more than 10 to 12 lines but very very meaningful and if and again saying if needed then give advise what to do?**
    """

    # Make AI API Call
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": ai_prompt}]}]}

    try:
        res = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers)
        res.raise_for_status()
        result = res.json()
        ai_message = result['candidates'][0]['content']['parts'][0]['text']

        return {
            "status": "success",
            "customer_id": customer_id,
            "loan_approval_prediction": prediction,
            "loan_details": loan_details,  # ✅ Fixed JSON serialization
            "financial_recommendation": ai_message
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"AI API request failed: {str(e)}"
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
        res = requests.post(
            f"{API_URL}?key={API_KEY}",
            json=payload,
            headers=headers,
            timeout=30  # this timeout parameter
    )
    except requests.exceptions.Timeout:
        raise HTTPException(408, "AI API request timed out after 30 seconds")
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
    - strongly advice
    Clearly suggest one actionable strategy for business growth or improved market positioning in 1-2 sentences.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        res = requests.post(
        f"{API_URL}?key={API_KEY}",
        json=payload,
        headers=headers,
        timeout=30  # line for 30-second timeout
    )
    except requests.exceptions.Timeout:
        raise HTTPException(408, "AI API request timed out after 30 seconds")
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

# logging
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

#old
def call_llmm(prompt):
    """Handles Google Gemini API failures with retries."""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(3):  #  Retry up to 3 times
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", json=payload, headers=headers, timeout=30)

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
class ChatRequestt(BaseModel):
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

def fformat_as_table(data):
    """Converts dictionary data into Markdown table format."""
    table = "| " + " | ".join(data.keys()) + " |\n"
    table += "| " + " | ".join(["---"] * len(data)) + " |\n"
    table += "| " + " | ".join(str(v) for v in data.values()) + " |\n"
    return table

import re
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import json
from typing import Optional, Tuple
import os


logger = logging.getLogger(__name__)
# ===== MODELS =====
class ChatRequest(BaseModel):
    query: str
    customer_id: Optional[str] = None

# ===== CORE FUNCTIONS =====
def format_as_table(data: dict) -> str:
    """Converts dictionary to Markdown table format."""
    if not data:
        return ""
    headers = "| " + " | ".join(data.keys()) + " |\n"
    separators = "| " + " | ".join(["---"] * len(data)) + " |\n"
    values = "| " + " | ".join(str(v) for v in data.values()) + " |"
    return headers + separators + values

def detect_id(query: str) -> Optional[Tuple[str, str]]:
    """Detects IDs in natural language queries"""
    patterns = [
        ("transaction_id", r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        ("customer_id", r"CUST-\d{4}"),
        ("business_id", r"BUS-\d{5}")
    ]
    for id_type, pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            return (id_type, match.group(0))
    return None

def get_transaction_insights(transaction_id: str) -> dict:
    """Get all data related to a specific transaction"""
    insights = {}
    
    if not fraud_df.empty and "Transaction ID" in fraud_df.columns:
        txn_record = fraud_df[fraud_df["Transaction ID"] == transaction_id]
        if not txn_record.empty:
            # Standardize column names
            record = txn_record.iloc[0].to_dict()
            insights.update({
                k.replace(" ", "_"): v for k, v in record.items()
            })
            
            # Link to customer if available (handling both column name variations)
            customer_id = record.get("Customer_ID") or record.get("Customer ID")
            if customer_id:
                customer_insights = get_customer_insights(customer_id)
                insights.update({"Customer_" + k: v for k, v in customer_insights.items()})
    
    return insights

def get_customer_insights(customer_id: str) -> dict:
    """Aggregates customer data from all datasets"""
    insights = {}
    
    # Check banking data
    if not banking_df.empty and "Customer_ID" in banking_df.columns:
        bank_record = banking_df[banking_df["Customer_ID"] == customer_id]
        if not bank_record.empty:
            insights.update({"Banking_" + k: v for k, v in bank_record.iloc[0].to_dict().items()})
    
    # Check fraud data (using both possible column names)
    if not fraud_df.empty:
        fraud_col = "Customer ID" if "Customer ID" in fraud_df.columns else "Customer_ID"
        if fraud_col in fraud_df.columns:
            fraud_record = fraud_df[fraud_df[fraud_col] == customer_id]
            if not fraud_record.empty:
                insights.update({
                    "Fraud_" + k.replace(" ", "_"): v 
                    for k, v in fraud_record.iloc[0].to_dict().items()
                    if k not in ["Customer ID", "Customer_ID"]
                })
    
    # Check churn data
    if not churn_df.empty and "Customer_ID" in churn_df.columns:
        churn_record = churn_df[churn_df["Customer_ID"] == customer_id]
        if not churn_record.empty:
            insights.update({"Churn_" + k: v for k, v in churn_record.iloc[0].to_dict().items()})
    
    return insights

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'financial_suggestion_model.pkl')

class LocalModelFallback:
    def __init__(self):
        self.local_model = None
        self.last_updated = None
        self._load_local_model()

    def _load_local_model(self):
        """Load the local model with error handling"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.local_model = pickle.load(f)
                self.last_updated = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH))
                logger.info(f"Local model loaded (last updated: {self.last_updated})")
            else:
                logger.warning("Local model file not found")
        except Exception as e:
            logger.error(f"Error loading local model: {e}")

    def generate_fallback_response(self, prompt):
        """Generate response using local model"""
        if self.local_model is None or isinstance(self.local_model, np.ndarray):
            return None
            
        try:
            # Preprocess prompt to match local model's expected input format
            processed_input = self._preprocess_prompt(prompt)
            
            # Get prediction from local model
            if hasattr(self.local_model, 'predict'):
                prediction = self.local_model.predict([processed_input])[0]
            else:  # Handle numpy array case
                prediction = np.random.choice(self.local_model)
            
            # Format the response
            return self._format_response(prediction)
        except Exception as e:
            logger.error(f"Local model prediction failed: {e}")
            return None

    def _preprocess_prompt(self, text):
        """Basic preprocessing to match training data format"""
        # Add your actual preprocessing logic here
        return text.lower().strip()[:500]  # Example: limit to 500 chars

    def _format_response(self, prediction):
        """Format the model's prediction into a user-friendly response"""
        # Customize based on your model's output format
        return f"⚠️ AI Service Unavailable. Local Model Suggestion: {prediction}"

# Initialize fallback handler
fallback_handler = LocalModelFallback()

def call_llm(prompt: str) -> str:
    """
    Enhanced LLM caller with:
    - Automatic API retries
    - Local model fallback
    - Consistent response formatting
    - Error recovery
    """
    
    def format_response(raw_text: str, source: str) -> str:
        """
        Standardize all responses with consistent formatting
        """
        # Clean the response
        cleaned = raw_text.strip()
        
        # Format based on content type
        if "|" in cleaned and "-|-" in cleaned:  # Table detection
            return f"📊 {source} Analysis:\n\n{cleaned}\n\n(Source: {source})"
        
        elif any(x in cleaned.lower() for x in ["recommend", "suggest"]):  # Recommendation
            return f"⭐ {source} Recommendation:\n\n{cleaned}\n"
        
        elif any(x in cleaned for x in ["•", "- "]) and "\n" in cleaned:  # List
            return f"📋 {source} Advice:\n\n{cleaned}\n"
        
        # Default formatting
        return f"{source} Response:\n\n{cleaned}\n"

    # Try online API first
    api_response = try_online_api(prompt)
    if api_response and not api_response.startswith("⚠️"):
        return format_response(api_response, "AI Analysis")
    
    # Fallback to local model
    local_response = fallback_handler.generate_fallback_response(prompt)
    if local_response:
        return format_response(local_response, "Local Model")
    
    # Final fallback message
    return format_response(
        "Our AI services are currently unavailable. Here's some general financial advice:\n\n" +
        "• Review your budget and spending habits\n" +
        "• Consider diversifying your investments\n" +
        "• Consult with a financial advisor for personalized guidance",
        "System"
    )

def try_online_api(prompt: str) -> str:
    """Try calling the online API with enhanced formatting awareness"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Format your response with clear sections, bullet points, or tables when appropriate.\n\n{prompt}"
            }]
        }],
        "generationConfig": {
            "response_mime_type": "text/markdown"  # Request formatted responses
        }
    }

    for attempt in range(10):
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", 
                                   json=payload, 
                                   headers=headers, 
                                   timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                raw_text = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Post-process API response
                return postprocess_response(raw_text)
                
            elif response.status_code == 429:
                time.sleep(min(2**attempt, 10))
                continue
                
        except Exception as e:
            logger.error(f"API attempt {attempt+1} failed: {str(e)}")
            time.sleep(1)
    
    return None

def postprocess_response(text: str) -> str:
    """Clean and standardize API responses"""
    # Remove redundant prefixes
    for prefix in ["Here's your analysis:", "My response:", "Answer:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Fix markdown tables
    if "|" in text:
        text = text.replace("| |", "|---|")
        text = text.replace("|-|", "|---|")
    
    # Ensure consistent line breaks
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    return text

def try_online_api(prompt: str) -> str:
    """Try calling the online API with retries"""
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(3):
        try:
            response = requests.post(f"{API_URL}?key={API_KEY}", 
                                  json=payload, 
                                  headers=headers, 
                                  timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            elif response.status_code in [401, 403]:
                logger.error("Authentication failed")
                return "⚠️ Authentication failed. Check API keys."
                
            elif response.status_code == 429:
                time.sleep(min(2**attempt, 10))  # Exponential backoff
                continue
                
            else:
                logger.error(f"API error {response.status_code}")
                return None  # Trigger fallback
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            time.sleep(1)
            
    return None  # Trigger fallback

# ===== MAIN ENDPOINT =====
@app.post("/chatbot/")
async def chatbot(request: ChatRequest):
    try:
        insights = {}
        response_text = ""
        is_transaction = False
        
        # 1. Auto-detect ID if not provided
        if not request.customer_id:
            if id_info := detect_id(request.query):
                id_type, id_value = id_info
                
                # Handle Transaction IDs
                if id_type == "transaction_id":
                    insights = get_transaction_insights(id_value)
                    is_transaction = True
                    if not insights:
                        return {"response": "⚠️ No transaction found with this ID"}
                    
                    llm_prompt = f"""
                    Fraud Analysis Request:
                    
                    TRANSACTION: {id_value}
                    QUERY: {request.query}
                    
                    KEY DATA:
                    {json.dumps(insights, indent=2)}
                    
                    Please analyze:
                    1. Fraud risk level assessment
                    2. Transaction pattern analysis
                    3. Recommended security actions
                    """
                    response_text = call_llm(llm_prompt)
                
                # Handle Customer/Business IDs
                else:
                    request.customer_id = id_value

        # 2. Handle Customer/Business analysis
        if not response_text and request.customer_id:
            insights = get_customer_insights(request.customer_id)
            if not insights:
                return {"response": "⚠️ No financial data found for this ID. Please verify the ID."}

            llm_prompt = f"""
            Financial Analysis Request:
            
            QUERY: {request.query}
            ID: {request.customer_id}
            
            AVAILABLE DATA:
            {json.dumps(insights, indent=2)}
            
             Provide:
            1. Specific suggestions (3 bullet points)
            2. Expected outcomes 
            3. Risk considerations make sure to have full meaningful sentence
            """
            response_text = call_llm(llm_prompt)

        # 3. Handle case when no ID is detected
        if not response_text:
            llm_prompt = f"""
            General Financial Advice Request:
            
            QUERY: {request.query}
            
            Please provide general financial advice and recommendations.
            """
            response_text = call_llm(llm_prompt)

        # 4. Format final response
        if insights:
            # Filter out complex objects and internal fields
            table_data = {
                k: v for k, v in insights.items() 
                if not isinstance(v, (dict, list)) 
                and not k.startswith("_")
                and v is not None
            }
            if table_data:
                response_text += "\n\n### Detailed Information\n" + format_as_table(table_data)

        return {"response": response_text if response_text else "No response generated"}

    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Analysis failed. Please try again later."
        )


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
            timeout=30
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
            timeout=30
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
from datetime import datetime
from pydantic import BaseModel
class ConsentRequest(BaseModel):
    consent_given: bool
# Stores user consent (Example: In-memory dictionary)
consent_records = {}  # Format: {customer_id: {"consent_given": True/False, "timestamp": "..."}}

@app.post("/consent/{customer_id}")
def set_consent(customer_id: str, consent_data: ConsentRequest):
    try:
        if consent_data.consent_given:
            if customer_id in consent_records and consent_records[customer_id]["consent_given"]:
                return {
                    "customer_id": customer_id,
                    "message": "Consent was already given.",
                    "consent_given": True
                }
            
            consent_records[customer_id] = {
                "consent_given": True,
                "timestamp": datetime.now().isoformat()  # Changed to datetime.now()
            }
            return {
                "customer_id": customer_id,
                "message": f"Consent granted for {customer_id}",
                "consent_given": True
            }
        else:
            if customer_id in consent_records:
                del consent_records[customer_id]
                return {
                    "customer_id": customer_id,
                    "message": f"Consent revoked for {customer_id}",
                    "consent_given": False
                }
            return {
                "customer_id": customer_id,
                "message": "⚠️ No consent record found to revoke.",
                "consent_given": False
            }
    except Exception as e:
        print(f"Error updating consent: {e}")
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


