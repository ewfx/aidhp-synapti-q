import pandas as pd
import random

# Function to generate market insights data
def generate_market_insights(num_records=10000):
    data = []
    
    for i in range(1, num_records + 1):
        business_id = f"MKT-{str(i).zfill(5)}"
        
        # Market Insights
        market_trend_score = random.randint(1, 10)  # 1-10 score
        industry_growth_rate = round(random.uniform(-5, 15), 2)  # -5% to 15% growth
        competitor_count = random.randint(5, 500)  # Number of competitors
        customer_acquisition_cost = round(random.uniform(5, 500), 2)  # Cost per customer
        customer_retention_rate = round(random.uniform(50, 95), 2)  # Retention % (50-95%)
        advertising_budget = round(random.uniform(5000, 5000000), 2)  # Ad spend
        brand_reputation_score = round(random.uniform(1, 5), 2)  # Score (1-5)
        social_media_engagement = random.randint(100, 1000000)  # Engagement volume
        sentiment_score = round(random.uniform(1, 5), 2)  # Sentiment rating
        
        # Store data
        data.append([
            business_id, market_trend_score, industry_growth_rate, competitor_count,
            customer_acquisition_cost, customer_retention_rate, advertising_budget,
            brand_reputation_score, social_media_engagement, sentiment_score
        ])
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=[
        "business_id", "market_trend_score", "industry_growth_rate", "competitor_count",
        "customer_acquisition_cost", "customer_retention_rate", "advertising_budget",
        "brand_reputation_score", "social_media_engagement", "sentiment_score"
    ])
    
    # Save to CSV
    df.to_csv("market_insights.csv", index=False)
    print("✅ Market insights dataset created: market_insights.csv")

# Run the function to generate dataset
generate_market_insights()
