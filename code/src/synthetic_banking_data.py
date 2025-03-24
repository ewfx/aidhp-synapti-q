import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

# Number of records to generate
num_records = 500

# Generate synthetic data
data = {
    "Customer_ID": [fake.uuid4() for _ in range(num_records)],
    "Name": [fake.name() for _ in range(num_records)],
    "Age": np.random.randint(18, 70, num_records),
    "Income": np.random.randint(30000, 200000, num_records),
    "Loan_Amount": np.random.randint(5000, 50000, num_records),
    "Credit_Score": np.random.randint(300, 850, num_records),
    "Debit_Card_Usage": np.random.randint(0, 100, num_records),
    "Credit_Card_Usage": np.random.randint(0, 100, num_records),
    "Auto_Loan_Eligibility": np.random.choice(["Yes", "No"], num_records)
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv("synthetic_banking_data.csv", index=False)

print("Synthetic banking dataset generated successfully!")
