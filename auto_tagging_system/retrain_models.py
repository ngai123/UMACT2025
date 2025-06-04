import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import re
from datetime import datetime

# Load configuration
with open('umbank_integrated_classifier/model_configuration.json', 'r') as f:
    config = json.load(f)

# Load and preprocess data
print("Loading data...")
df = pd.read_csv('cleaned_UMBank_complaints_data.csv')
print(f"Loaded {len(df)} complaints")

# Clean and prepare data
df['Complaint'] = df['Complaint'].astype(str).str.lower()
df['Complaint'] = df['Complaint'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
df['Complaint'] = df['Complaint'].apply(lambda x: re.sub(r'x{2,}', '', x))
df = df[df['Complaint'].str.len() >= config['model_config']['min_complaint_length']]

# Create strategic groups
product_to_group = {}
for group, products in config['strategic_groups'].items():
    for product in products:
        product_to_group[product.lower()] = group

df['Strategic_Group'] = df['Product'].str.lower().map(product_to_group)
df = df.dropna(subset=['Strategic_Group'])

# Prepare encoders
strategic_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['strategic_labels'] = strategic_encoder.fit_transform(df['Strategic_Group'])
df['product_labels'] = product_encoder.fit_transform(df['Product'])

# Remove product classes with fewer than 2 samples
product_counts = df['Product'].value_counts()
valid_products = product_counts[product_counts >= 2].index
initial_rows = len(df)
df = df[df['Product'].isin(valid_products)]
removed_rows = initial_rows - len(df)
if removed_rows > 0:
    print(f"Removed {removed_rows} samples from rare product classes (<2 samples)")

# Split data
X = df['Complaint'].values
y_strategic = df['strategic_labels'].values
y_product = df['product_labels'].values

X_train, X_test, y_strategic_train, y_strategic_test = train_test_split(
    X, y_strategic, test_size=config['model_config']['test_size'], 
    random_state=42, stratify=y_strategic
)

_, _, y_product_train, y_product_test = train_test_split(
    X, y_product, test_size=config['model_config']['test_size'],
    random_state=42, stratify=y_product
)

# Create and train strategic model
print("\nTraining strategic model...")
strategic_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=config['model_parameters']['tfidf']['max_features'],
        ngram_range=tuple(config['model_parameters']['tfidf']['ngram_range']),
        stop_words=config['model_parameters']['tfidf']['stop_words']
    )),
    ('clf', LogisticRegression(
        C=config['model_parameters']['logistic_regression']['C'],
        solver=config['model_parameters']['logistic_regression']['solver'],
        class_weight=config['model_parameters']['logistic_regression']['class_weight'],
        random_state=config['model_parameters']['logistic_regression']['random_state']
    ))
])

strategic_pipeline.fit(X_train, y_strategic_train)

# Create and train product model
print("\nTraining product model...")
product_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=config['model_parameters']['tfidf']['max_features'],
        ngram_range=tuple(config['model_parameters']['tfidf']['ngram_range']),
        stop_words=config['model_parameters']['tfidf']['stop_words']
    )),
    ('clf', LogisticRegression(
        C=config['model_parameters']['logistic_regression']['C'],
        solver=config['model_parameters']['logistic_regression']['solver'],
        class_weight=config['model_parameters']['logistic_regression']['class_weight'],
        random_state=config['model_parameters']['logistic_regression']['random_state']
    ))
])

product_pipeline.fit(X_train, y_product_train)

# Evaluate models
print("\nEvaluating models...")
strategic_pred = strategic_pipeline.predict(X_test)
product_pred = product_pipeline.predict(X_test)

strategic_accuracy = accuracy_score(y_strategic_test, strategic_pred)
product_accuracy = accuracy_score(y_product_test, product_pred)

print(f"Strategic Model Accuracy: {strategic_accuracy:.4f}")
print(f"Product Model Accuracy: {product_accuracy:.4f}")

# Save models and encoders
print("\nSaving models and encoders...")
joblib.dump(strategic_pipeline, 'umbank_integrated_classifier/strategic_model_pipeline.joblib')
joblib.dump(product_pipeline, 'umbank_integrated_classifier/product_model_pipeline.joblib')
joblib.dump(strategic_encoder, 'umbank_integrated_classifier/strategic_label_encoder.joblib')
joblib.dump(product_encoder, 'umbank_integrated_classifier/product_label_encoder.joblib')

# Generate confusion matrices
print("\nGenerating confusion matrices...")
plt.figure(figsize=(12, 8))
strategic_cm = confusion_matrix(y_strategic_test, strategic_pred)
sns.heatmap(strategic_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=strategic_encoder.classes_,
            yticklabels=strategic_encoder.classes_)
plt.title('Strategic Groups - Confusion Matrix')
plt.xlabel('Predicted Strategic Group')
plt.ylabel('Actual Strategic Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('umbank_integrated_classifier/strategic_confusion_matrix.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(15, 12))
product_cm = confusion_matrix(y_product_test, product_pred)
sns.heatmap(product_cm, annot=False, cmap='Blues')
plt.title('Product Categories - Confusion Matrix')
plt.savefig('umbank_integrated_classifier/product_confusion_matrix.png', dpi=300, bbox_inches='tight')

print("\nModel retraining completed successfully!") 