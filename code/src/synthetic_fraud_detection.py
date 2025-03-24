import pandas as pd
import random
import uuid
from datetime import datetime, timedelta

# Generate synthetic fraud detection dataset
num_records = 10000

def generate_transaction_id():
    return str(uuid.uuid4())

def generate_customer_id():
    return f"CUSTFRAUD-{random.randint(1000, 9999)}"

def generate_transaction_amount():
    return round(random.uniform(10, 10000), 2)

def generate_transaction_type():
    return random.choice(["Online Purchase", "ATM Withdrawal", "Wire Transfer", "POS Payment", "Crypto Exchange"])

def generate_location():
    return random.choice(["USA", "UK", "Canada", "Germany", "India", "Australia", "France", "Singapore", "Japan", "UAE"])

def generate_device_used():
    return random.choice(["Mobile", "Desktop", "Tablet", "Unknown"])

def generate_transaction_time():
    start_date = datetime(2024, 1, 1)
    random_days = random.randint(0, 90)
    random_seconds = random.randint(0, 86400)  # Full day in seconds
    return start_date + timedelta(days=random_days, seconds=random_seconds)

def generate_historical_fraud_flag():
    return random.choice([0, 1])  # 0 = No history, 1 = Previously flagged for fraud

def generate_risk_score():
    return round(random.uniform(0, 1), 2)  # Probability score from 0 (low risk) to 1 (high risk)

def generate_fraud_label(risk_score, historical_fraud_flag):
    if risk_score > 0.85 or (historical_fraud_flag == 1 and risk_score > 0.7):
        return 1  # Fraudulent
    return 0  # Legitimate

# Create dataset
data = []
for _ in range(num_records):
    transaction_id = generate_transaction_id()
    customer_id = generate_customer_id()
    amount = generate_transaction_amount()
    trans_type = generate_transaction_type()
    location = generate_location()
    device = generate_device_used()
    trans_time = generate_transaction_time()
    fraud_history = generate_historical_fraud_flag()
    risk_score = generate_risk_score()
    fraud_label = generate_fraud_label(risk_score, fraud_history)
    
    data.append([
        transaction_id, customer_id, amount, trans_type, location, device,
        trans_time, fraud_history, risk_score, fraud_label
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "Transaction ID", "Customer ID", "Transaction Amount", "Transaction Type",
    "Location", "Device Used", "Transaction Time", "Historical Fraud Flag",
    "Risk Score", "Fraud Label"
])

# Save to CSV
df.to_csv("synthetic_fraud_detection.csv", index=False)

print("Synthetic fraud detection dataset generated and saved as 'synthetic_fraud_detection.csv'.")
