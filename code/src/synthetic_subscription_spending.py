import pandas as pd
import random
from faker import Faker

fake = Faker()

# Define categories for subscriptions & spending
subscription_services = ["Netflix", "Spotify", "Amazon Prime", "Disney+", "Apple Music", "Hulu", "PlayStation Plus", "Adobe Creative Cloud"]
spending_categories = ["Food", "Entertainment", "Shopping", "Travel", "Healthcare", "Utilities", "Education"]

# Generate synthetic data
data = []
num_records = 10000  # Adjust number of records if needed

for _ in range(num_records):
    customer_id = f"CUSTS-{random.randint(1000, 9999)}"
    subscription = random.choice(subscription_services)
    monthly_fee = round(random.uniform(5, 30), 2)
    spending_category = random.choice(spending_categories)
    monthly_spending = round(random.uniform(50, 500), 2)
    cancel_probability = round(random.uniform(0, 1), 2)  # Probability of canceling subscription

    data.append([
        customer_id, subscription, monthly_fee, spending_category, monthly_spending, cancel_probability
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=["Customer_ID", "Subscription", "Monthly_Fee", "Spending_Category", "Monthly_Spending", "Cancel_Probability"])

# Save to CSV
df.to_csv("subscription_spending_patterns.csv", index=False)

print("✅ Subscription & Spending Patterns dataset generated: subscription_spending_patterns.csv")
