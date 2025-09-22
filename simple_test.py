#!/usr/bin/env python3
"""
Simple test script for text classification

Tests the basic functionality without Unicode characters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import re
import warnings

warnings.filterwarnings('ignore')

def clean_text_simple(text):
    """Simple text cleaning function"""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Simple Text Classification Test")
    print("=" * 40)

    # Load dataset
    try:
        df = pd.read_csv('dataset/text classifcation.csv')
        print(f"Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Get columns
    text_col = df.columns[0]
    target_col = df.columns[1]

    print(f"Text column: {text_col}")
    print(f"Target column: {target_col}")
    print(f"Categories: {df[target_col].unique()}")

    # Clean texts
    print("Cleaning texts...")
    df['cleaned_text'] = df[text_col].apply(clean_text_simple)
    df_clean = df[df['cleaned_text'].str.strip() != '']

    print(f"Texts after cleaning: {len(df_clean)}")

    # Prepare data
    X = df_clean['cleaned_text']
    y = df_clean[target_col]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Create features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Feature matrix shape: {X_train_tfidf.shape}")

    # Train model
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("\nTest completed successfully!")
    return 0
'''main execution starts here
    if __name__ == "__main__":
        sys.exit(main())'''
if __name__ == "__main__":
    main()