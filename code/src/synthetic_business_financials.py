import pandas as pd
import random

# Define industries
industries = ["Retail", "Technology", "Healthcare", "Finance", "Manufacturing", "Real Estate", "Education"]

# Function to generate business financial data
def generate_business_financials(num_records=10000):
    data = []
    
    for i in range(1, num_records + 1):
        business_id = f"BUS-{str(i).zfill(5)}"
        business_name = f"Business_{i}"
        industry = random.choice(industries)
        
        # Financials
        annual_revenue = round(random.uniform(500000, 50000000), 2)  # Revenue between $500K and $50M
        annual_expenses = round(random.uniform(100000, annual_revenue * 0.9), 2)  # Expenses up to 90% of revenue
        profit_margin = round(((annual_revenue - annual_expenses) / annual_revenue) * 100, 2)  # Profit %
        
        # Credit & Loan
        credit_score = random.randint(300, 850)  # Credit score range
        loan_amount = round(random.uniform(0, annual_revenue * 0.3), 2)  # Loan up to 30% of revenue
        risk_category = "High" if credit_score < 500 else "Medium" if credit_score < 700 else "Low"
        
        # Social Media & Sentiment
        social_media_presence = random.randint(1000, 5000000)  # Followers
        customer_sentiment_score = round(random.uniform(1, 5), 2)  # Score between 1 and 5
        
        # Store data
        data.append([
            business_id, business_name, industry, annual_revenue, annual_expenses, profit_margin,
            credit_score, loan_amount, risk_category, social_media_presence, customer_sentiment_score
        ])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "business_id", "business_name", "industry", "annual_revenue", "annual_expenses", "profit_margin",
        "credit_score", "loan_amount", "risk_category", "social_media_presence", "customer_sentiment_score"
    ])
    
    # Save to CSV
    df.to_csv("business_financials.csv", index=False)
    print("✅ Business financial dataset created: business_financials.csv")

# Run the function to generate dataset
generate_business_financials()
