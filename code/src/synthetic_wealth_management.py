import pandas as pd
import random
import numpy as np

# Generate synthetic wealth management data
def generate_wealth_management_data(num_records=10000):
    customer_ids = [f"CUSTW-{str(i).zfill(5)}" for i in range(1, num_records + 1)]
    
    # Investment Risk Appetite Levels
    risk_levels = ["Low", "Moderate", "High"]
    
    # Asset classes
    asset_classes = ["Stocks", "Bonds", "Mutual Funds", "Real Estate", "Cryptocurrency", "Gold", "Savings"]
    
    # Generate Data
    data = []
    for cust_id in customer_ids:
        net_worth = round(random.uniform(5000, 5000000), 2)  # USD
        annual_income = round(random.uniform(30000, 500000), 2)
        risk_appetite = random.choice(risk_levels)
        primary_investment = random.choice(asset_classes)
        secondary_investment = random.choice([x for x in asset_classes if x != primary_investment])
        savings_percentage = round(random.uniform(5, 50), 2)  # % of income saved
        debt_to_income_ratio = round(random.uniform(0, 1), 2)  # 0 to 1 scale
        retirement_fund = round(random.uniform(1000, 1000000), 2)
        preferred_advisor = random.choice(["Robo-Advisor", "Human Financial Advisor", "None"])
        
        data.append([
            cust_id, net_worth, annual_income, risk_appetite, primary_investment,
            secondary_investment, savings_percentage, debt_to_income_ratio,
            retirement_fund, preferred_advisor
        ])
    
    # Create DataFrame
    columns = [
        "Customer_ID", "Net_Worth", "Annual_Income", "Risk_Appetite",
        "Primary_Investment", "Secondary_Investment", "Savings_Percentage",
        "Debt_to_Income_Ratio", "Retirement_Fund", "Preferred_Advisor"
    ]
    df = pd.DataFrame(data, columns=columns)
    
    return df

# Generate and save dataset
df_wealth = generate_wealth_management_data()
df_wealth.to_csv("synthetic_wealth_management.csv", index=False)
print("Synthetic Wealth Management dataset created successfully!")
