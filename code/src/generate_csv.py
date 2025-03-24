from fastapi import FastAPI, HTTPException
import pandas as pd
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Generate synthetic customer data
def generate_data(n=10000):
    data = []
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
    merchants = ["Amazon", "Flipkart", "Myntra", "Big Bazaar", "Reliance Digital", "DMart"]
    platforms = ["Facebook", "Instagram", "Twitter", "LinkedIn"]
    
    for i in range(n):
        customer = {
            "Customer_ID": f"CUST-{i+1:04d}",
            "Age": random.randint(18, 75),
            "Gender": random.choice(["Male", "Female"]),
            "Income": random.randint(300000, 2500000),
            "Location": random.choice(cities),
            "Last_Transaction": random.randint(100, 10000),
            "Frequent_Merchant": random.choice(merchants),
            "Average_Monthly_Spend": random.randint(5000, 100000),
            "Social_Media_Platform": random.choice(platforms),
            "Engagement_Score": random.randint(1, 100),
            "Customer_Feedback": random.choice(["Positive", "Neutral", "Negative"]),
            "Review_Score": random.randint(1, 5),
            "Credit_Score": random.randint(300, 850),
            "Existing_Loans": random.randint(0, 5),
            "Loan_Offer": random.choice(["Home Loan", "Car Loan", "Personal Loan", "Credit Card Offer"]),
            "Food_Expense": random.randint(500, 20000),
            "Entertainment_Expense": random.randint(500, 15000),
            "Bills_Expense": random.randint(1000, 50000),
            "Savings_Expense": random.randint(1000, 50000),
            "Fraud_Score": random.randint(0, 100),
            "Financial_Advice": random.choice(["Invest in Mutual Funds", "Increase Savings", "Reduce Spending", "Consider a Loan"]),
            "Investment_Profile": random.choice(["Stocks", "Bonds", "Fixed Deposits", "Gold"])
        }
        data.append(customer)
    
    df = pd.DataFrame(data)
    df.to_csv("synthetic_banking_data.csv", index=False)
    return df

# Generate dataset
df = generate_data()

# Encode categorical data for ML model
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])
le_platform = LabelEncoder()
df["Social_Media_Platform"] = le_platform.fit_transform(df["Social_Media_Platform"])

# Train AI Model for Loan Recommendation
X = df[["Age", "Income", "Credit_Score", "Existing_Loans", "Average_Monthly_Spend"]]
y = df["Loan_Offer"]
loan_model = RandomForestClassifier()
loan_model.fit(X, y)

print("Synthetic dataset generated and saved successfully!")