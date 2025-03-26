import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)

# Configuration
DATASET_PATH = os.path.join('Datasets', 'synthetic_banking_data.csv')
MODEL_SAVE_PATH = 'financial_suggestion_model.pkl'
PROTECTED_ATTRIBUTES = ['Gender', 'Location']

def analyze_sentiment(text):
    """Sentiment analysis using NLTK's VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(str(text))
    polarity = scores['compound'] * 0.5
    subjectivity = (1 - abs(scores['compound'])) * 0.5
    return polarity, subjectivity

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Validate required columns
    required_columns = {
        'Age', 'Income', 'Credit_Score',
        'Customer_Feedback', 'Last_Transaction'
    }
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate total expenses if not present
    expense_cols = ['Food_Expense', 'Entertainment_Expense', 'Bills_Expense']
    if all(col in df.columns for col in expense_cols):
        df['Total_Expenses'] = df[expense_cols].sum(axis=1)
    elif 'Total_Expenses' not in df.columns:
        raise ValueError("No expense columns found to calculate total expenses")
    
    return df

def generate_suggestion_labels(df):
    """Generate target labels based on financial patterns"""
    conditions = [
        (df['Credit_Score'] >= 750) & (df['Income'] > df['Total_Expenses'] * 1.5),
        (df['Credit_Score'] < 600) | (df['Total_Expenses'] > df['Income']),
        (df['Savings_Expense']/df['Income'] < 0.1) if 'Savings_Expense' in df.columns else False
    ]
    
    choices = [
        'Consider investment opportunities',
        'Focus on debt reduction',
        'Increase savings rate'
    ]
    
    df['Financial_Suggestion'] = np.select(conditions, choices, default='Maintain current strategy')
    return df

def apply_debiasing(df):
    """Apply debiasing techniques"""
    df = df.copy()
    
    # Normalize income by location if available
    if 'Location' in df.columns:
        location_means = df.groupby('Location')['Income'].transform('mean')
        global_mean = df['Income'].mean()
        df['Income'] = df['Income'] / location_means * global_mean
    
    return df

def train_suggestion_model():
    try:
        # 1. Load and preprocess data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data()
        
        # 2. Generate suggestion labels
        print("Generating financial suggestions...")
        df = generate_suggestion_labels(df)
        
        # 3. Apply debiasing
        print("Applying debiasing techniques...")
        df = apply_debiasing(df)
        
        # 4. Feature Engineering
        print("Engineering features...")
        # Handle dates
        df['Last_Transaction'] = pd.to_datetime(df['Last_Transaction'], errors='coerce')
        df = df.dropna(subset=['Last_Transaction'])
        df['Days_Since_Transaction'] = (pd.to_datetime('today') - df['Last_Transaction']).dt.days
        
        # Calculate financial ratios
        df['Income_Adjusted'] = df['Income'].replace(0, 1)
        df['Expense_Ratio'] = df['Total_Expenses'] / df['Income_Adjusted']
        
        if 'Savings_Expense' in df.columns:
            df['Savings_Ratio'] = df['Savings_Expense'] / df['Income_Adjusted']
        
        # Sentiment analysis
        sentiment_results = df['Customer_Feedback'].apply(
            lambda x: analyze_sentiment(x) if pd.notna(x) else (0.0, 0.5)
        )
        df[['Sentiment_Polarity', 'Sentiment_Subjectivity']] = pd.DataFrame(
            sentiment_results.tolist(), index=df.index
        )
        
        # 5. Prepare final features
        features = [
            'Age', 'Income', 'Days_Since_Transaction',
            'Expense_Ratio', 'Credit_Score',
            'Sentiment_Polarity', 'Sentiment_Subjectivity'
        ]
        
        if 'Savings_Ratio' in df.columns:
            features.append('Savings_Ratio')
        
        # 6. Encode categoricals if present
        categorical_cols = ['Gender', 'Location']
        label_encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
                features.append(col)
        
        # 7. Train/test split
        X = df[features]
        y = df['Financial_Suggestion']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 8. Train model
        print("Training suggestion model...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            random_state=42,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        
        # 9. Evaluate
        print("\nModel Evaluation:")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # 10. Save model
        artifacts = {
            'model': model,
            'features': features,
            'label_encoders': label_encoders,
            'suggestion_categories': list(y.unique()),
            'debiasing_info': {
                'protected_attributes': PROTECTED_ATTRIBUTES,
                'techniques_applied': [
                    'Income Normalization',
                    'Class Weight Balancing'
                ]
            }
        }
        joblib.dump(artifacts, MODEL_SAVE_PATH)
        print(f"\nModel saved to: {MODEL_SAVE_PATH}")
        
        # 11. Plot feature importance
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.title("Feature Importances for Financial Suggestions")
        plt.bar(range(len(features)), importances[indices], align="center")
        plt.xticks(range(len(features)), np.array(features)[indices], rotation=45)
        plt.tight_layout()
        plt.savefig('suggestion_feature_importance.png')
        print("Saved feature importance plot")
        
        return model
        
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_suggestion_model()