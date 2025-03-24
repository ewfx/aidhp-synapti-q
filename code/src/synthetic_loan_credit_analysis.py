import pandas as pd
import random
import uuid

def generate_loan_credit_data(num_rows=10000):
    data = []
    
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed']
    loan_purposes = ['Home', 'Car', 'Education', 'Personal', 'Business']
    approval_statuses = ['Approved', 'Rejected', 'Pending']
    risk_levels = ['Low', 'Medium', 'High']
    
    for _ in range(num_rows):
        customer_id = str(uuid.uuid4())[:8]
        age = random.randint(18, 70)
        income = random.randint(20000, 150000)
        employment_status = random.choice(employment_types)
        credit_score = random.randint(300, 850)
        existing_loans = random.randint(0, 5)
        loan_amount_requested = random.randint(5000, 500000)
        loan_purpose = random.choice(loan_purposes)
        loan_approval_status = random.choices(
            approval_statuses, 
            weights=[0.6, 0.3, 0.1],  # More chances for approval
            k=1
        )[0]
        debt_to_income_ratio = round(random.uniform(0.1, 0.8), 2)
        loan_interest_rate = round(random.uniform(2.5, 15.0), 2)
        loan_term = random.choice([12, 24, 36, 48, 60, 120, 180, 240, 360])  # Months
        default_risk = random.choices(risk_levels, weights=[0.5, 0.3, 0.2], k=1)[0]
        
        data.append([
            customer_id, age, income, employment_status, credit_score, 
            existing_loans, loan_amount_requested, loan_purpose, 
            loan_approval_status, debt_to_income_ratio, loan_interest_rate, 
            loan_term, default_risk
        ])
    
    columns = [
        "customer_id", "age", "income", "employment_status", "credit_score", 
        "existing_loans", "loan_amount_requested", "loan_purpose", 
        "loan_approval_status", "debt_to_income_ratio", "loan_interest_rate", 
        "loan_term", "default_risk"
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("loan_credit_analysis.csv", index=False)
    print("✅ Loan & Credit Score Analysis dataset created successfully!")

# Run the function
generate_loan_credit_data()
