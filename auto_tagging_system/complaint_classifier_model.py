# complaint_classifier_model.py
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

warnings.filterwarnings('ignore')

np.random.seed(42)

class SimpleComplaintClassifierLite:
    """
    Simpler Complaint to Product Classifier using TF-IDF and Logistic Regression.
    Uses ONLY: Complaint text → Product category
    """

    def __init__(self, csv_file_path=None, max_rows=None): # csv_file_path can be None if not training
        """
        Initialize the classifier

        Args:
            csv_file_path (str, optional): Path to the complaints CSV file. Required for training.
            max_rows (int): Maximum number of rows to use for training (None for all data)
        """
        self.csv_file_path = csv_file_path
        self.max_rows = max_rows
        self.model = None # This will be a scikit-learn pipeline
        self.label_encoder = None
        self.training_stats = {}
        self.target_column_name = None # Will be set during preprocessing

    def load_and_preprocess_data(self, use_grouped_products=True):
        """
        Load and preprocess ONLY complaint text and product data
        """
        if not self.csv_file_path or not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found at {self.csv_file_path}. Please provide a valid path for training.")

        if self.max_rows:
            df = pd.read_csv(self.csv_file_path, nrows=self.max_rows, sep=',')
            print(f"Loaded dataset shape (limited to {self.max_rows} rows): {df.shape}")
        else:
            df = pd.read_csv(self.csv_file_path, sep=',')
            print(f"Loaded full dataset shape: {df.shape}")

        print(f"Original columns: {list(df.columns)}")

        required_columns = ['Complaint', 'Product']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        df = df[required_columns].copy()
        print(f"Using only columns: {list(df.columns)}")

        df = df.dropna(subset=['Complaint', 'Product'])
        print(f"After removing missing values: {df.shape}")

        if use_grouped_products:
            df = self.create_product_groups(df) # This method now also sets self.target_column_name
            self.target_column_name = 'Product_Group' # Ensure it's set after grouping
        else:
            self.target_column_name = 'Product'

        # Clean complaint text (more basic cleaning)
        df['Complaint'] = df['Complaint'].astype(str).str.lower()
        df['Complaint'] = df['Complaint'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
        df['Complaint'] = df['Complaint'].apply(lambda x: re.sub(r'x{2,}', '', x))

        df = df[df['Complaint'].str.len() >= 10]
        print(f"After removing short complaints: {df.shape}")

        class_counts = df[self.target_column_name].value_counts()
        rare_classes = class_counts[class_counts < 3].index.tolist()

        if rare_classes:
            print(f"\nFiltering {len(rare_classes)} rare classes with <3 samples.")
            df = df[~df[self.target_column_name].isin(rare_classes)]
            print(f"After filtering rare classes: {df.shape}")

        self.label_encoder = LabelEncoder()
        df['labels'] = self.label_encoder.fit_transform(df[self.target_column_name])

        print(f"\n{'='*50}")
        print(f"FINAL DATASET SUMMARY")
        print(f"{'='*50}")
        print(f"Total samples: {len(df)}")
        print(f"Number of product categories: {len(self.label_encoder.classes_)}")
        print(f"Target distribution:\n{df[self.target_column_name].value_counts().head()}")

        self.training_stats.update({
            'total_samples': len(df),
            'num_classes': len(self.label_encoder.classes_),
            'class_distribution': df[self.target_column_name].value_counts().to_dict(),
            'features_used': ['Complaint (TF-IDF)'],
            'target': self.target_column_name # self.target_column_name should be set by now
        })

        df = df[df['Complaint'].str.strip().astype(bool)]
        print(f"After removing any empty complaints post-cleaning: {df.shape}")

        if df.empty:
            raise ValueError("No data left after preprocessing. Check cleaning steps or input data.")

        return df[['Complaint', 'labels', self.target_column_name]]

    def create_product_groups(self, df):
        """ Group products into main categories and remove products not in defined groups """
        product_groups = {
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
            'Banking Services': [
                'Bank account or service'
            ]
        }
        # Consider case-insensitivity if products in CSV might vary in casing
        product_to_group = {p.lower(): g for g, ps in product_groups.items() for p in ps}
        
        # Apply mapping. Ensure 'Product' column is also lowercased for matching if necessary
        # For robustness, handle if 'Product' column is not string type or has NaN
        df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning if df is a slice
        df_copy['Product_Group'] = df_copy['Product'].astype(str).str.lower().map(product_to_group)

        initial_rows = len(df_copy)
        df_copy.dropna(subset=['Product_Group'], inplace=True)
        removed_rows = initial_rows - len(df_copy)

        if removed_rows > 0:
            print(f"\nRemoved {removed_rows} samples whose products did not map to a defined group.")

        print(f"Shape after product grouping and removing unmapped products: {df_copy.shape}")

        if df_copy.empty:
            raise ValueError("No data left after product grouping. Check product_groups against the CSV's 'Product' column values & casing.")

        print(f"Product Grouping Applied: Now using {df_copy['Product_Group'].nunique()} main categories.")
        print(f"Product Group distribution:\n{df_copy['Product_Group'].value_counts()}")
        self.target_column_name = 'Product_Group' # Set target column name here
        return df_copy

    def create_train_test_split(self, df, test_size=0.2):
        X = df['Complaint'].values
        y = df['labels'].values

        # Ensure there are enough samples in each class for stratification if multiple classes exist
        min_samples_per_class = 2 
        if len(np.unique(y)) > 1 and (len(y) < min_samples_per_class * len(np.unique(y)) or np.bincount(y).min() < min_samples_per_class):
            stratify_split = False
            print("Warning: Insufficient samples for stratified splitting (each class needs at least 2 samples). Using non-stratified split.")
        elif len(np.unique(y)) == 1: # Only one class
            stratify_split = False
            print("Warning: Only one class present. Using non-stratified split.")
        else:
            stratify_split = True
            
        if len(y) < 2 : # Not enough samples to split at all meaningfully for train/test
             print("Warning: Very few samples (<2). Cannot create a meaningful train/test split. Using all for training, none for testing.")
             return {'train': {'texts': X, 'labels': y}, 'test': {'texts': np.array([]), 'labels': np.array([])}}


        if stratify_split:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError as e: # Catch potential errors if stratification fails for other reasons
                print(f"Stratified split failed with error: {e}. Falling back to non-stratified split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=None
                )
        else: # Non-stratified split
             X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=None
            )
        
        print(f"\nData split: Train: {len(X_train)}, Test: {len(X_test)} samples")
        return {'train': {'texts': X_train, 'labels': y_train},
                'test': {'texts': X_test, 'labels': y_test}}

    def train_model(self, data_splits, output_dir='./simple_complaint_classifier_lite'):
        X_train = data_splits['train']['texts']
        y_train = data_splits['train']['labels']

        if len(X_train) == 0:
            raise ValueError("Cannot train model: No training data provided.")

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
            ('clf', LogisticRegression(random_state=42, solver='liblinear', C=1.0, class_weight='balanced'))
        ])

        print(f"\nStarting training with TF-IDF and Logistic Regression...")
        print(f"Training on {len(X_train)} samples")

        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        end_time = datetime.now()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, 'model_pipeline.joblib')
        label_encoder_path = os.path.join(output_dir, 'label_encoder.joblib')
        training_stats_path = os.path.join(output_dir, 'training_stats.json')

        joblib.dump(self.model, model_path)
        joblib.dump(self.label_encoder, label_encoder_path)

        # Update training_stats with info that should be known by now
        self.training_stats.update({
            'training_time_seconds': (end_time - start_time).total_seconds(),
            'training_date': datetime.now().isoformat(),
            'model_type': 'TF-IDF + LogisticRegression',
            'model_pipeline_path': model_path,
            'label_encoder_path': label_encoder_path,
            'target': self.target_column_name # ensure this is correct
        })

        with open(training_stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2, default=str)

        print(f"Training completed and model saved to {output_dir}!")
        return self.model

    def evaluate_model(self, data_splits):
        if not self.model:
            raise Exception("Model not trained yet. Call train_model first.")

        X_test = data_splits['test']['texts']
        y_true = data_splits['test']['labels']

        if len(X_test) == 0:
            print("No test data to evaluate. Skipping evaluation.")
            return {
                'accuracy': 0, 'f1_macro': 0, 'f1_weighted': 0,
                'predictions': np.array([]), 'true_labels': y_true, 
                'target_names': [], 'classification_report': {},
                'valid_labels_for_report': []
            }
        
        print("\nEvaluating model on test set...")
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        unique_labels_in_test_and_pred = np.unique(np.concatenate((y_true, y_pred)))
        
        report_target_names = []
        valid_labels_for_report = [] # numeric labels
        for label_idx in unique_labels_in_test_and_pred:
            if label_idx < len(self.label_encoder.classes_):
                report_target_names.append(self.label_encoder.classes_[label_idx])
                valid_labels_for_report.append(label_idx)
            else:
                print(f"Warning: Label index {label_idx} from predictions/test data is out of bounds for label_encoder classes (size: {len(self.label_encoder.classes_)}). Skipping in report.")
        
        # Ensure valid_labels_for_report is sorted for consistent report ordering
        # and that report_target_names corresponds to this order
        sorted_indices_for_report = np.argsort(valid_labels_for_report)
        valid_labels_for_report = np.array(valid_labels_for_report)[sorted_indices_for_report].tolist()
        report_target_names = np.array(report_target_names)[sorted_indices_for_report].tolist()


        report_str = classification_report(
            y_true, y_pred,
            labels=valid_labels_for_report,
            target_names=report_target_names,
            zero_division=0
        )
        report_dict = classification_report(
            y_true, y_pred,
            labels=valid_labels_for_report,
            target_names=report_target_names,
            output_dict=True,
            zero_division=0
        )

        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS (TF-IDF + Logistic Regression)")
        print(f"{'='*60}")
        print(f"Test samples: {len(y_true)}")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"Number of classes in report (present in test/pred): {len(report_target_names)}")
        print(f"\nDetailed Classification Report:\n{report_str}")

        return {
            'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
            'predictions': y_pred, 'true_labels': y_true, 
            'target_names': report_target_names, # String names of classes in the report
            'classification_report': report_dict,
            'valid_labels_for_report': valid_labels_for_report # Numeric labels in the report
        }

    def plot_confusion_matrix(self, y_true, y_pred, class_labels_for_plot_ticks, save_path=None):
        if len(y_true) == 0 and len(y_pred) == 0:
            print("Skipping confusion matrix: No true labels or predictions.")
            return

        # Determine the set of labels to use for the confusion matrix.
        # These should be all numeric labels that were actually present in y_true or y_pred.
        # This ensures the matrix dimensions match the data.
        present_numeric_labels = np.unique(np.concatenate((y_true, y_pred)))
        
        # `class_labels_for_plot_ticks` (results['target_names']) are the string names for these present_numeric_labels
        # We need to ensure that `plot_tick_names` corresponds to `present_numeric_labels` in the correct order.
        # The `confusion_matrix` `labels` argument dictates the rows/columns.
        
        # Map present_numeric_labels to their string names for ticks
        # The `class_labels_for_plot_ticks` already comes from results['target_names']
        # which should correspond to unique labels in test/pred, sorted.
        # So, `class_labels_for_plot_ticks` should be the tick labels if `present_numeric_labels` are sorted.
        
        # Sort present_numeric_labels to ensure consistent order for cm and tick labels
        sorted_present_numeric_labels = np.sort(present_numeric_labels)
        
        # Generate tick names based on sorted_present_numeric_labels
        # This relies on self.label_encoder being correctly fitted.
        plot_tick_names = [self.label_encoder.classes_[i] for i in sorted_present_numeric_labels if i < len(self.label_encoder.classes_)]


        cm = confusion_matrix(y_true, y_pred, labels=sorted_present_numeric_labels)

        plt.figure(figsize=(max(8, int(len(plot_tick_names) * 0.9)), max(6, int(len(plot_tick_names) * 0.7))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=plot_tick_names, yticklabels=plot_tick_names)
        
        title = (f'Confusion Matrix\nComplaint Text → {self.target_column_name or "Product Group"}\n'
                 f'{self.training_stats.get("total_samples", "N/A")} training samples')
        plt.title(title)
        plt.xlabel('Predicted Product Group')
        plt.ylabel('Actual Product Group')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        plt.show() # Keep for interactive training runs

    def predict_complaint(self, complaint_text, model_dir='./simple_complaint_classifier_lite'):
        if not self.model or not self.label_encoder:
            model_path = os.path.join(model_dir, 'strategic_model_pipeline.joblib')
            label_encoder_path = os.path.join(model_dir, 'strategic_label_encoder.joblib')
            training_stats_path = os.path.join(model_dir, 'training_statistics.json')

            print(f"[DEBUG] Attempting to load model from: {model_path}")
            print(f"[DEBUG] Attempting to load label encoder from: {label_encoder_path}")
            print(f"[DEBUG] Attempting to load training stats from: {training_stats_path}")

            if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
                print(f"[ERROR] Model or label encoder file not found.")
                raise FileNotFoundError(
                    f"Model ({model_path}) or LabelEncoder ({label_encoder_path}) not found in {model_dir}. "
                    "Please train the model first."
                )
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(label_encoder_path)
            print(f"[DEBUG] Model loaded: {self.model is not None}")
            print(f"[DEBUG] Label encoder loaded: {self.label_encoder is not None}")
            try:
                with open(training_stats_path, 'r') as f:
                    stats = json.load(f)
                    self.target_column_name = stats.get('target', 'Product_Group')
            except FileNotFoundError:
                print(f"Warning: {training_stats_path} not found. Target column name might be default.")
                self.target_column_name = 'Product_Group'

        cleaned_text = re.sub(r'\s+', ' ', complaint_text.lower().strip())
        cleaned_text = re.sub(r'x{2,}', '', cleaned_text)
        print(f"[DEBUG] Cleaned input: '{cleaned_text}'")

        if not cleaned_text:
            print("[ERROR] Input is empty after cleaning.")
            return {
                'predicted_product': "N/A - Empty input",
                'confidence': 0.0,
                'all_probabilities': {}
            }

        probabilities = self.model.predict_proba([cleaned_text])[0]
        print(f"[DEBUG] Raw model probabilities: {probabilities}")

        class_names = self.label_encoder.classes_
        print(f"[DEBUG] Model class names: {class_names}")

        # When using the strategic model, the model's prediction is already the strategic group.
        # We don't need to re-map using the product_to_group logic.
        
        # Find the index of the highest probability
        predicted_class_idx = np.argmax(probabilities)
        
        # Get the strategic group name and its confidence
        predicted_group = class_names[predicted_class_idx]
        confidence = probabilities[predicted_class_idx]
        
        # Prepare the probabilities in a dictionary format (optional, but useful for UI)
        all_probabilities = dict(zip(class_names, probabilities))

        print(f"[DEBUG] Predicted group: {predicted_group}, Confidence: {confidence}")
        return {
            'predicted_product': predicted_group, # Renaming key for consistency, still represents the strategic group
            'confidence': float(confidence), # Ensure confidence is float
            'all_probabilities': all_probabilities
        }