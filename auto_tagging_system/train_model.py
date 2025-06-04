# -*- coding: utf-8 -*-
"""
UMBank Integrated Complaint Classifier - Strategic Groups + ML Prediction

This script combines strategic business analysis with machine learning to:
1. Use the strategic groupings identified from the full dataset analysis
2. Train classifiers for both strategic groups and individual products
3. Provide comprehensive prediction and business intelligence capabilities
4. Generate insights for operational decision-making
"""

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
from datetime import datetime
import warnings
import re
import json

# --- Enhanced Configuration with Strategic Groups ---
CONFIG = {
    'csv_file_path': r'C:\Users\User\Downloads\UMACT\cleaned_UMBank_complaints_data.csv',
    'max_rows': 27972,  # Full dataset
    'use_strategic_groups': True,  # NEW: Use strategic business groups
    'use_original_products': False,  # Option to train on individual products
    'test_size': 0.2,
    'output_dir': './umbank_integrated_classifier',
    'random_state': 42,
    'tfidf_max_features': 5000,
    'tfidf_ngram_range': (1, 2),
    'logistic_C': 1.0,
    'logistic_solver': 'liblinear',
    'min_complaint_length': 10,
    'train_dual_models': True  # Train both strategic and product-level models
}

# Strategic Groups Configuration Based on Full Dataset Analysis
STRATEGIC_GROUPS_CONFIG = {
    'High-Volume Consumer Products': [
        'Credit card or prepaid card',
        'Checking or savings account', 
        'Mortgage',
        'Credit reporting, credit repair services, or other personal consumer reports'
    ],
    'Credit & Lending Portfolio': [
        'Credit card',
        'Vehicle loan or lease',
        'Consumer Loan',
        'Student loan',
        'Payday loan, title loan, or personal loan',
        'Payday loan'
    ],
    'Regulatory & Collections': [
        'Debt collection',
        'Credit reporting'
    ],
    'Specialized Services': [
        'Money transfer, virtual currency, or money service',
        'Money transfers',
        'Other financial service',
        'Prepaid card'
    ],
    'Banking Services': [  # Additional group for any unmapped products
        'Bank account or service'
    ]
}

# Business Intelligence Metrics from Analysis
BUSINESS_METRICS = {
    'volume_thresholds': {
        'high': 2070,
        'medium_min': 180,
        'low_max': 180
    },
    'strategic_insights': {
        'High-Volume Consumer Products': {
            'total_complaints': 20280,
            'avg_per_product': 5070,
            'strategy': 'Dedicated high-capacity teams with streamlined processes',
            'priority': 'Critical'
        },
        'Credit & Lending Portfolio': {
            'total_complaints': 3112,
            'avg_per_product': 519,
            'strategy': 'Specialized lending team with risk management focus',
            'priority': 'High'
        },
        'Regulatory & Collections': {
            'total_complaints': 1365,
            'avg_per_product': 682,
            'strategy': 'Compliance-focused team with legal support',
            'priority': 'High'
        },
        'Specialized Services': {
            'total_complaints': 1565,
            'avg_per_product': 391,
            'strategy': 'Technical specialists for transaction-based services',
            'priority': 'Medium'
        }
    }
}

warnings.filterwarnings('ignore')
np.random.seed(CONFIG['random_state'])

# --- Enhanced Helper Functions ---

def print_enhanced_config(config, strategic_config, business_metrics):
    """Prints enhanced configuration with strategic insights."""
    print("=" * 70)
    print("🏦 UMBANK INTEGRATED COMPLAINT CLASSIFIER CONFIGURATION")
    print("=" * 70)
    print("Model Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n📊 Strategic Groups ({len(strategic_config)} groups):")
    for group, products in strategic_config.items():
        metrics = business_metrics['strategic_insights'].get(group, {})
        print(f"  🎯 {group}:")
        print(f"     Products: {len(products)}")
        print(f"     Expected Volume: {metrics.get('total_complaints', 'N/A')}")
        print(f"     Priority: {metrics.get('priority', 'N/A')}")
    print("-" * 70)

def load_and_validate_data(file_path, max_rows=None, random_state=42):
    """Enhanced data loading with validation against expected structure."""
    print(f"📂 Loading UMBank complaint data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded: {df.shape}")
        
        # Validate expected columns
        required_columns = ['Complaint', 'Product']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
            print(f"📊 Using {len(df)} rows (sampled from full dataset)")
        
        print(f"📈 Total complaints to process: {len(df):,}")
        return df
    
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found.")
        raise
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        raise

def clean_complaint_text_enhanced(df, min_length):
    """Enhanced complaint text cleaning with better preprocessing."""
    print(f"\n🧹 Cleaning complaint text (min length: {min_length} chars)...")
    
    if 'Complaint' not in df.columns:
        raise ValueError("'Complaint' column not found in DataFrame.")

    df_cleaned = df.copy()
    
    # Enhanced text cleaning
    df_cleaned['Complaint'] = df_cleaned['Complaint'].astype(str)
    df_cleaned['Complaint'] = df_cleaned['Complaint'].str.lower()
    df_cleaned['Complaint'] = df_cleaned['Complaint'].apply(
        lambda x: re.sub(r'\s+', ' ', x.strip())
    )
    df_cleaned['Complaint'] = df_cleaned['Complaint'].apply(
        lambda x: re.sub(r'x{2,}', '', x)  # Remove redactions
    )
    df_cleaned['Complaint'] = df_cleaned['Complaint'].apply(
        lambda x: re.sub(r'[^\w\s]', ' ', x)  # Remove special characters
    )
    
    # Remove short complaints
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Complaint'].str.len() >= min_length]
    removed_short = initial_rows - len(df_cleaned)
    
    # Remove empty complaints
    df_cleaned = df_cleaned[df_cleaned['Complaint'].str.strip().astype(bool)]
    final_rows = len(df_cleaned)
    
    print(f"   📉 Removed {removed_short} complaints shorter than {min_length} chars")
    print(f"   📋 Final dataset: {final_rows:,} complaints")
    
    if final_rows == 0:
        raise ValueError("No complaints remaining after cleaning.")
    
    # Show sample cleaned complaints
    print(f"\n🔍 Sample cleaned complaints:")
    for i, complaint in enumerate(df_cleaned['Complaint'].head(3)):
        print(f"   {i+1}. {complaint[:80]}...")
    
    return df_cleaned

def apply_strategic_grouping(df, strategic_groups_config):
    """Apply strategic business grouping to products."""
    print(f"\n🎯 Applying strategic business grouping...")
    
    # Create product to group mapping
    product_to_group = {}
    for group, products in strategic_groups_config.items():
        for product in products:
            product_to_group[product] = group
    
    df_grouped = df.copy()
    df_grouped['Strategic_Group'] = df_grouped['Product'].map(product_to_group)
    
    # Handle unmapped products
    unmapped = df_grouped['Strategic_Group'].isnull().sum()
    if unmapped > 0:
        print(f"   ⚠️  Warning: {unmapped} complaints with unmapped products")
        unmapped_products = df_grouped[df_grouped['Strategic_Group'].isnull()]['Product'].unique()
        print(f"   Unmapped products: {list(unmapped_products)}")
        
        # Remove unmapped for now (could assign to 'Other' group instead)
        df_grouped = df_grouped.dropna(subset=['Strategic_Group'])
    
    # Display strategic group distribution
    group_dist = df_grouped['Strategic_Group'].value_counts()
    print(f"\n📊 Strategic Group Distribution:")
    for group, count in group_dist.items():
        expected = BUSINESS_METRICS['strategic_insights'].get(group, {}).get('total_complaints', 'N/A')
        print(f"   🎯 {group}: {count:,} complaints (expected: {expected})")
    
    print(f"   📈 Total classified: {len(df_grouped):,} complaints")
    return df_grouped

def prepare_dual_targets(df, strategic_groups_config):
    """Prepare both strategic group and product-level targets."""
    print(f"\n🎯 Preparing dual classification targets...")
    
    # Apply strategic grouping
    df_strategic = apply_strategic_grouping(df, strategic_groups_config)
    
    # Prepare strategic group target
    strategic_encoder = LabelEncoder()
    df_strategic['strategic_labels'] = strategic_encoder.fit_transform(df_strategic['Strategic_Group'])
    
    # Prepare product-level target
    product_encoder = LabelEncoder() 
    df_strategic['product_labels'] = product_encoder.fit_transform(df_strategic['Product'])
    
    print(f"   ✅ Strategic groups: {len(strategic_encoder.classes_)} classes")
    print(f"   ✅ Product categories: {len(product_encoder.classes_)} classes")
    
    return df_strategic, strategic_encoder, product_encoder

def build_dual_models(config):
    """Build both strategic and product-level model pipelines."""
    print(f"\n🏗️  Building dual model pipelines...")
    
    # Strategic group model
    strategic_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            stop_words='english',
            ngram_range=config['tfidf_ngram_range']
        )),
        ('clf', LogisticRegression(
            random_state=config['random_state'],
            solver=config['logistic_solver'],
            C=config['logistic_C'],
            class_weight='balanced'
        ))
    ])
    
    # Product-level model (same architecture)
    product_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=config['tfidf_max_features'],
            stop_words='english',
            ngram_range=config['tfidf_ngram_range']
        )),
        ('clf', LogisticRegression(
            random_state=config['random_state'],
            solver=config['logistic_solver'],
            C=config['logistic_C'],
            class_weight='balanced'
        ))
    ])
    
    print(f"   ✅ Strategic group pipeline created")
    print(f"   ✅ Product-level pipeline created")
    
    return strategic_pipeline, product_pipeline

def train_dual_models(strategic_model, product_model, X_train, y_strategic_train, y_product_train):
    """Train both strategic and product-level models."""
    print(f"\n🎓 Training dual models...")
    start_time = datetime.now()
    
    # Train strategic model
    print(f"   📚 Training strategic group classifier...")
    strategic_model.fit(X_train, y_strategic_train)
    
    # Train product model
    print(f"   📚 Training product-level classifier...")
    product_model.fit(X_train, y_product_train)
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    print(f"   ✅ Dual training completed in {training_time:.2f} seconds")
    return strategic_model, product_model, training_time

def evaluate_dual_models(strategic_model, product_model, X_test, y_strategic_test, y_product_test, 
                        strategic_encoder, product_encoder, output_dir):
    """Evaluate both models and generate comprehensive reports."""
    print(f"\n📊 Evaluating dual models...")
    
    # Strategic model evaluation
    y_strategic_pred = strategic_model.predict(X_test)
    strategic_accuracy = accuracy_score(y_strategic_test, y_strategic_pred)
    strategic_f1 = f1_score(y_strategic_test, y_strategic_pred, average='weighted')
    
    # Product model evaluation  
    y_product_pred = product_model.predict(X_test)
    product_accuracy = accuracy_score(y_product_test, y_product_pred)
    product_f1 = f1_score(y_product_test, y_product_pred, average='weighted')
    
    print(f"\n🎯 STRATEGIC GROUP MODEL RESULTS:")
    print(f"   Accuracy: {strategic_accuracy:.4f} ({strategic_accuracy*100:.2f}%)")
    print(f"   F1-Score: {strategic_f1:.4f}")
    print(f"   Classes: {len(strategic_encoder.classes_)}")
    
    print(f"\n🎯 PRODUCT-LEVEL MODEL RESULTS:")
    print(f"   Accuracy: {product_accuracy:.4f} ({product_accuracy*100:.2f}%)")
    print(f"   F1-Score: {product_f1:.4f}")
    print(f"   Classes: {len(product_encoder.classes_)}")
    
    # Generate confusion matrices
    os.makedirs(output_dir, exist_ok=True)
    
    # Strategic confusion matrix
    strategic_cm = confusion_matrix(y_strategic_test, y_strategic_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(strategic_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=strategic_encoder.classes_, 
                yticklabels=strategic_encoder.classes_)
    plt.title('Strategic Groups - Confusion Matrix')
    plt.xlabel('Predicted Strategic Group')
    plt.ylabel('Actual Strategic Group')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    strategic_cm_path = os.path.join(output_dir, "strategic_confusion_matrix.png")
    plt.savefig(strategic_cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Product confusion matrix (top 10 products only for readability)
    product_cm = confusion_matrix(y_product_test, y_product_pred)
    if len(product_encoder.classes_) > 10:
        print(f"   📋 Product confusion matrix saved (too large to display: {len(product_encoder.classes_)} classes)")
        product_cm_path = os.path.join(output_dir, "product_confusion_matrix.png")
        plt.figure(figsize=(15, 12))
        sns.heatmap(product_cm, annot=False, cmap='Blues')
        plt.title('Product Categories - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(product_cm_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(product_cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=product_encoder.classes_,
                    yticklabels=product_encoder.classes_)
        plt.title('Product Categories - Confusion Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        product_cm_path = os.path.join(output_dir, "product_confusion_matrix.png")
        plt.savefig(product_cm_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    return {
        'strategic_accuracy': strategic_accuracy,
        'strategic_f1': strategic_f1,
        'product_accuracy': product_accuracy, 
        'product_f1': product_f1,
        'strategic_cm_path': strategic_cm_path,
        'product_cm_path': product_cm_path
    }

def save_integrated_artifacts(strategic_model, product_model, strategic_encoder, product_encoder, 
                            training_stats, output_dir):
    """Save all models and artifacts."""
    print(f"\n💾 Saving integrated classifier artifacts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    strategic_model_path = os.path.join(output_dir, "strategic_model_pipeline.joblib")
    product_model_path = os.path.join(output_dir, "product_model_pipeline.joblib") 
    strategic_encoder_path = os.path.join(output_dir, "strategic_label_encoder.joblib")
    product_encoder_path = os.path.join(output_dir, "product_label_encoder.joblib")
    
    joblib.dump(strategic_model, strategic_model_path)
    joblib.dump(product_model, product_model_path)
    joblib.dump(strategic_encoder, strategic_encoder_path)
    joblib.dump(product_encoder, product_encoder_path)
    
    # Save configuration and stats
    config_path = os.path.join(output_dir, "model_configuration.json")
    stats_path = os.path.join(output_dir, "training_statistics.json")
    
    # Enhanced configuration with strategic insights
    full_config = {
        'model_config': CONFIG,
        'strategic_groups': STRATEGIC_GROUPS_CONFIG,
        'business_metrics': BUSINESS_METRICS,
        'training_stats': training_stats
    }
    
    with open(config_path, 'w') as f:
        json.dump(full_config, f, indent=2, default=str)
        
    with open(stats_path, 'w') as f:
        json.dump(training_stats, f, indent=2, default=str)
    
    print(f"   ✅ Strategic model: {strategic_model_path}")
    print(f"   ✅ Product model: {product_model_path}")
    print(f"   ✅ Strategic encoder: {strategic_encoder_path}")
    print(f"   ✅ Product encoder: {product_encoder_path}")
    print(f"   ✅ Configuration: {config_path}")
    print(f"   ✅ Statistics: {stats_path}")
    
    return {
        'strategic_model_path': strategic_model_path,
        'product_model_path': product_model_path,
        'strategic_encoder_path': strategic_encoder_path,
        'product_encoder_path': product_encoder_path,
        'config_path': config_path,
        'stats_path': stats_path
    }

class IntegratedComplaintPredictor:
    """Integrated predictor that provides both strategic and product-level predictions."""
    
    def __init__(self, strategic_model, product_model, strategic_encoder, product_encoder):
        self.strategic_model = strategic_model
        self.product_model = product_model
        self.strategic_encoder = strategic_encoder
        self.product_encoder = product_encoder
    
    def predict_complaint(self, complaint_text):
        """Predict both strategic group and specific product for a complaint."""
        # Clean the input
        cleaned_text = self._clean_text(complaint_text)
        
        # Strategic prediction
        strategic_probs = self.strategic_model.predict_proba([cleaned_text])[0]
        strategic_pred_idx = np.argmax(strategic_probs)
        strategic_group = self.strategic_encoder.classes_[strategic_pred_idx]
        strategic_confidence = strategic_probs[strategic_pred_idx]
        
        # Product prediction
        product_probs = self.product_model.predict_proba([cleaned_text])[0]
        product_pred_idx = np.argmax(product_probs)
        product = self.product_encoder.classes_[product_pred_idx]
        product_confidence = product_probs[product_pred_idx]
        
        # Get business insights
        business_insight = BUSINESS_METRICS['strategic_insights'].get(strategic_group, {})
        
        return {
            'strategic_prediction': {
                'group': strategic_group,
                'confidence': strategic_confidence,
                'strategy': business_insight.get('strategy', 'N/A'),
                'priority': business_insight.get('priority', 'N/A')
            },
            'product_prediction': {
                'product': product,
                'confidence': product_confidence
            },
            'recommendation': self._get_routing_recommendation(strategic_group, product, strategic_confidence)
        }
    
    def _clean_text(self, text):
        """Clean input text to match training preprocessing."""
        cleaned = str(text).lower().strip()
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'x{2,}', '', cleaned)
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        return cleaned
    
    def _get_routing_recommendation(self, strategic_group, product, confidence):
        """Provide routing recommendation based on predictions."""
        if confidence > 0.8:
            confidence_level = "High"
        elif confidence > 0.6:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
            
        insights = BUSINESS_METRICS['strategic_insights'].get(strategic_group, {})
        priority = insights.get('priority', 'Medium')
        
        return {
            'confidence_level': confidence_level,
            'routing_priority': priority,
            'suggested_team': strategic_group,
            'escalation_needed': confidence < 0.6 or priority == 'Critical'
        }

def run_integrated_demo(predictor):
    """Run demonstration of the integrated prediction system."""
    print(f"\n" + "="*70)
    print(f"🚀 INTEGRATED COMPLAINT PREDICTION DEMO")
    print(f"="*70)
    
    demo_complaints = [
        "My credit card was charged twice for the same transaction and I need this fixed immediately",
        "I can't access my online banking account and this is really frustrating",
        "The ATM ate my card and won't give it back to me",
        "I applied for a mortgage but haven't heard back for weeks",
        "There are unauthorized charges on my account statement again",
        "Debt collector keeps calling me at work even though I don't owe this money",
        "My student loan payment was not processed correctly this month",
        "Someone used my identity to open a bank account without my permission",
        "The money transfer I sent never arrived at the destination",
        "I'm having trouble with my vehicle loan payment process"
    ]
    
    for i, complaint in enumerate(demo_complaints, 1):
        print(f"\n📝 Complaint {i}:")
        print(f"   Text: {complaint}")
        
        try:
            result = predictor.predict_complaint(complaint)
            
            print(f"   🎯 Strategic Group: {result['strategic_prediction']['group']}")
            print(f"   📊 Confidence: {result['strategic_prediction']['confidence']:.2%}")
            print(f"   🏷️  Specific Product: {result['product_prediction']['product']}")
            print(f"   📈 Product Confidence: {result['product_prediction']['confidence']:.2%}")
            print(f"   🎪 Strategy: {result['strategic_prediction']['strategy']}")
            print(f"   ⚡ Priority: {result['strategic_prediction']['priority']}")
            print(f"   🎯 Routing: {result['recommendation']['suggested_team']}")
            print(f"   🚨 Escalation Needed: {result['recommendation']['escalation_needed']}")
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
        
        print("-" * 50)

def display_integrated_summary(training_stats, file_paths):
    """Display comprehensive summary of the integrated system."""
    print(f"\n" + "="*70)
    print(f"🎉 UMBANK INTEGRATED CLASSIFIER - TRAINING COMPLETE")
    print(f"="*70)
    
    print(f"📊 DATASET SUMMARY:")
    print(f"   • Total samples processed: {training_stats.get('total_samples', 'N/A'):,}")
    print(f"   • Training samples: {training_stats.get('training_samples', 'N/A'):,}")
    print(f"   • Test samples: {training_stats.get('test_samples', 'N/A'):,}")
    print(f"   • Training time: {training_stats.get('training_time', 0):.2f} seconds")
    
    print(f"\n🎯 MODEL PERFORMANCE:")
    print(f"   Strategic Groups Model:")
    print(f"     • Accuracy: {training_stats.get('strategic_accuracy', 0):.2%}")
    print(f"     • F1-Score: {training_stats.get('strategic_f1', 0):.4f}")
    print(f"     • Classes: {training_stats.get('strategic_classes', 'N/A')}")
    
    print(f"   Product-Level Model:")
    print(f"     • Accuracy: {training_stats.get('product_accuracy', 0):.2%}")
    print(f"     • F1-Score: {training_stats.get('product_f1', 0):.4f}")
    print(f"     • Classes: {training_stats.get('product_classes', 'N/A')}")
    
    print(f"\n💼 BUSINESS INSIGHTS:")
    for group, metrics in BUSINESS_METRICS['strategic_insights'].items():
        print(f"   🎯 {group}:")
        print(f"     • Expected Volume: {metrics['total_complaints']:,}")
        print(f"     • Strategy: {metrics['strategy']}")
        print(f"     • Priority: {metrics['priority']}")
    
    print(f"\n📁 SAVED ARTIFACTS:")
    for key, path in file_paths.items():
        print(f"   • {key.replace('_', ' ').title()}: {path}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Load models using joblib.load() for production use")
    print(f"   2. Use IntegratedComplaintPredictor for real-time predictions")
    print(f"   3. Implement routing logic based on strategic group predictions")
    print(f"   4. Monitor model performance and retrain periodically")
    print(f"   5. Integrate with customer service workflow systems")

# --- Main Execution ---
def main():
    """Main function to run the integrated complaint classification system."""
    
    print_enhanced_config(CONFIG, STRATEGIC_GROUPS_CONFIG, BUSINESS_METRICS)
    
    # 1. Load and prepare data
    raw_df = load_and_validate_data(CONFIG['csv_file_path'], CONFIG.get('max_rows'), CONFIG['random_state'])
    df = raw_df[['Complaint', 'Product']].copy()
    
    # 2. Clean and preprocess
    df_cleaned = clean_complaint_text_enhanced(df, CONFIG['min_complaint_length'])
    
    # 3. Prepare dual targets
    df_final, strategic_encoder, product_encoder = prepare_dual_targets(df_cleaned, STRATEGIC_GROUPS_CONFIG)
    
    # 4. Prepare features and split data
    X = df_final['Complaint']
    y_strategic = df_final['strategic_labels']
    y_product = df_final['product_labels']
    
    X_train, X_test, y_strategic_train, y_strategic_test, y_product_train, y_product_test = train_test_split(
        X, y_strategic, y_product, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y_strategic  # Stratify on strategic groups
    )
    
    print(f"\n📊 Data Split Summary:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing: {len(X_test):,} samples")
    
    # 5. Build and train models
    strategic_model, product_model = build_dual_models(CONFIG)
    strategic_model, product_model, training_time = train_dual_models(
        strategic_model, product_model, X_train, y_strategic_train, y_product_train
    )
    
    # 6. Evaluate models
    evaluation_results = evaluate_dual_models(
        strategic_model, product_model, X_test, y_strategic_test, y_product_test,
        strategic_encoder, product_encoder, CONFIG['output_dir']
    )
    
    # 7. Prepare training statistics
    training_stats = {
        'training_date': datetime.now().isoformat(),
        'total_samples': len(df_final),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'strategic_classes': len(strategic_encoder.classes_),
        'product_classes': len(product_encoder.classes_),
        'training_time': training_time,
        **evaluation_results
    }
    
    # 8. Save artifacts
    file_paths = save_integrated_artifacts(
        strategic_model, product_model, strategic_encoder, product_encoder,
        training_stats, CONFIG['output_dir']
    )
    
    # 9. Create integrated predictor and run demo
    predictor = IntegratedComplaintPredictor(
        strategic_model, product_model, strategic_encoder, product_encoder
    )
    run_integrated_demo(predictor)
    
    # 10. Display final summary
    display_integrated_summary(training_stats, file_paths)
    
    return predictor, training_stats, file_paths

if __name__ == "__main__":
    try:
        predictor, stats, paths = main()
        print(f"\n✅ Integration successful! Predictor ready for use.")
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()