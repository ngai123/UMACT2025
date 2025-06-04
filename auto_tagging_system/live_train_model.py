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
from complaint_classifier_model import SimpleComplaintClassifierLite

class LiveTrainComplaintClassifier(SimpleComplaintClassifierLite):
    """
    Enhanced version of SimpleComplaintClassifierLite that supports live training
    with business intelligence integration
    """
    
    # Business Intelligence Metrics
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
            },
            'Banking Services': {
                'total_complaints': 1000,
                'avg_per_product': 1000,
                'strategy': 'Standard banking service team',
                'priority': 'Medium'
            }
        }
    }
    
    def __init__(self, csv_file_path=None, max_rows=None, model_dir='./complaint_classifier_model'):
        """
        Initialize the live training classifier
        
        Args:
            csv_file_path (str, optional): Path to the complaints CSV file
            max_rows (int, optional): Maximum number of rows to use for training
            model_dir (str): Directory to save/load model files
        """
        super().__init__(csv_file_path, max_rows)
        self.model_dir = model_dir
        self.training_history = []
        self.business_metrics = self.BUSINESS_METRICS
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
    def live_train(self, new_data, test_size=0.2):
        """
        Train the model with new data and update the existing model
        
        Args:
            new_data (pd.DataFrame): New data to train on
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training statistics
        """
        print("\nStarting live training with new data...")
        
        # Preprocess new data
        new_data = new_data[['Complaint', 'Product']].copy()
        new_data = self.clean_complaint_text(new_data, min_length=10)
        
        if new_data.empty:
            raise ValueError("No valid data after preprocessing")
            
        # Prepare target column
        new_data, _ = self.prepare_target_column(new_data, True, self.PRODUCT_GROUPS_CONFIG)
        
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            new_data['labels'] = self.label_encoder.fit_transform(new_data[self.target_column_name])
        else:
            # Update label encoder with new classes if any
            new_classes = set(new_data[self.target_column_name].unique())
            existing_classes = set(self.label_encoder.classes_)
            
            if new_classes - existing_classes:
                print(f"New classes found: {new_classes - existing_classes}")
                # Combine old and new classes
                all_classes = sorted(list(existing_classes | new_classes))
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(all_classes)
            
            new_data['labels'] = self.label_encoder.transform(new_data[self.target_column_name])
        
        # Split data
        data_splits = self.create_train_test_split(new_data, test_size)
        
        # Train model
        if self.model is None:
            # First time training
            self.train_model(data_splits, self.model_dir)
        else:
            # Update existing model
            X_train = data_splits['train']['texts']
            y_train = data_splits['train']['labels']
            
            print(f"\nUpdating model with {len(X_train)} new samples...")
            start_time = datetime.now()
            self.model.fit(X_train, y_train)
            end_time = datetime.now()
            
            # Save updated model
            model_path = os.path.join(self.model_dir, 'model_pipeline.pkl')
            label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            
            joblib.dump(self.model, model_path)
            joblib.dump(self.label_encoder, label_encoder_path)
            
            # Update training stats
            self.training_stats.update({
                'last_update': datetime.now().isoformat(),
                'new_samples': len(X_train),
                'update_time_seconds': (end_time - start_time).total_seconds()
            })
            
            # Save updated stats
            stats_path = os.path.join(self.model_dir, 'training_stats.json')
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2, default=str)
        
        # Evaluate model
        eval_results = self.evaluate_model(data_splits)
        
        # Save training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': len(new_data),
            'accuracy': eval_results['accuracy'],
            'f1_macro': eval_results['f1_macro']
        })
        
        return eval_results
    
    def get_training_history(self):
        """Return the training history"""
        return self.training_history
    
    def plot_training_history(self, save_path=None):
        """Plot the training history metrics"""
        if not self.training_history:
            print("No training history available")
            return
            
        history_df = pd.DataFrame(self.training_history)
        
        plt.figure(figsize=(12, 6))
        plt.plot(history_df['timestamp'], history_df['accuracy'], label='Accuracy', marker='o')
        plt.plot(history_df['timestamp'], history_df['f1_macro'], label='F1 Macro', marker='s')
        
        plt.title('Model Performance Over Time')
        plt.xlabel('Training Time')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def predict_complaint(self, complaint_text):
        """Enhanced prediction with business insights"""
        prediction = super().predict_complaint(complaint_text)
        
        # Add business insights
        strategic_group = prediction['predicted_product']
        business_insight = self.business_metrics['strategic_insights'].get(strategic_group, {})
        
        prediction.update({
            'business_insights': {
                'strategy': business_insight.get('strategy', 'N/A'),
                'priority': business_insight.get('priority', 'N/A'),
                'avg_complaints': business_insight.get('avg_per_product', 'N/A')
            }
        })
        
        return prediction

# Example usage
if __name__ == "__main__":
    # Initialize the live training classifier
    live_classifier = LiveTrainComplaintClassifier(
        csv_file_path='cleaned_UMBank_complaints_data.csv',
        model_dir='./complaint_classifier_model'
    )
    
    # Load and preprocess initial data
    initial_data = live_classifier.load_and_preprocess_data(use_grouped_products=True)
    
    # Initial training
    initial_results = live_classifier.live_train(initial_data)
    print("\nInitial training results:", initial_results)
    
    # Example of live training with new data
    # new_data = pd.DataFrame({
    #     'Complaint': ['New complaint text here'],
    #     'Product': ['Product category']
    # })
    # update_results = live_classifier.live_train(new_data)
    # print("\nUpdate results:", update_results) 