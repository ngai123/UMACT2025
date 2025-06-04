# app.py
from flask import Flask, request, jsonify, render_template
import os
from complaint_classifier_model import SimpleComplaintClassifierLite
from bert import DualModelSentimentAnalyzer
import pandas as pd
import re

app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = r'C:\Users\User\Downloads\UMACT\umbank_integrated_classifier'

# --- Load Model ---
classifier = None 
sentiment_analyzer = None
try:
    # Initialize with the new model directory
    classifier = SimpleComplaintClassifierLite(csv_file_path=None)
    sentiment_analyzer = DualModelSentimentAnalyzer()
    print("Attempting to pre-load model artifacts...")
    model_path = 'umbank_integrated_classifier/strategic_model_pipeline.joblib'
    label_encoder_path = 'umbank_integrated_classifier/strategic_label_encoder.joblib'
    if os.path.exists(model_path):
        classifier.predict_complaint("Initialize model load", model_dir=MODEL_DIR)
        print(f"Model artifacts (model, label encoder, stats) should be loaded from {MODEL_DIR}.")
        if classifier.model and classifier.label_encoder:
            print("Classifier components (model, label encoder) seem ready.")
        else:
            print("Warning: Classifier components might not have loaded correctly during pre-load. Will attempt on first API call.")
    else:
        print(f"Warning: Model pipeline file not found at {model_path}. "
              "The app will likely fail on predict. Please ensure the model files are in the correct directory.")
except Exception as e:
    print(f"Error during initialization: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global classifier 
    global sentiment_analyzer

    if classifier is None: 
        print("CRITICAL: Classifier object is None. Attempting to re-initialize for this request.")
        try:
            classifier = SimpleComplaintClassifierLite(csv_file_path=None)
        except Exception as e:
            print(f"CRITICAL: Failed to re-initialize classifier: {e}")
            return jsonify({'error': f'Classifier not available: {str(e)}'}), 500
    if sentiment_analyzer is None:
        try:
            sentiment_analyzer = DualModelSentimentAnalyzer()
        except Exception as e:
            print(f"CRITICAL: Failed to initialize sentiment analyzer: {e}")
            return jsonify({'error': f'Sentiment analyzer not available: {str(e)}'}), 500

    try:
        data = request.get_json()
        if 'complaint_text' not in data:
            return jsonify({'error': 'Missing "complaint_text" in request'}), 400
        
        complaint_text = data['complaint_text']
        if not isinstance(complaint_text, str) or not complaint_text.strip():
             return jsonify({'error': 'Complaint text must be a non-empty string'}), 400

        # Product prediction
        prediction_result = classifier.predict_complaint(complaint_text, model_dir=MODEL_DIR)
        # Sentiment analysis
        sentiment_result = sentiment_analyzer.analyze_sentiment(complaint_text)

        # Merge results
        response = {
            **prediction_result,
            'sentiment': {
                'mean_score': sentiment_result['mean_score'],
                'sentiment': sentiment_result['sentiment'],
                'emoji': sentiment_result['emoji'],
                'model1_label': sentiment_result['model1_result']['label'],
                'model1_score': sentiment_result['model1_result']['score'],
                'model2_label': sentiment_result['model2_result']['label'],
                'model2_score': sentiment_result['model2_result']['score']
            }
        }
        return jsonify(response)

    except FileNotFoundError as fnf_error:
        print(f"Prediction Error - File Not Found: {str(fnf_error)}")
        return jsonify({'error': f"Model files not found. Ensure the model is trained. Details: {str(fnf_error)}"}), 500
    except ValueError as val_error:
        print(f"Prediction Error - Value Error: {str(val_error)}")
        return jsonify({'error': f"Error during prediction: {str(val_error)}"}), 400
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An unexpected error occurred on the server: {str(e)}'}), 500

# Add new endpoint for live training
@app.route('/train', methods=['POST'])
def train():
    global classifier
    
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized'}), 500
        
    try:
        data = request.get_json()
        if 'complaints' not in data or not isinstance(data['complaints'], list):
            return jsonify({'error': 'Missing or invalid "complaints" array in request'}), 400
            
        # Convert complaints to DataFrame
        new_data = pd.DataFrame(data['complaints'])
        
        # Validate required columns
        required_columns = ['Complaint', 'Product']
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            return jsonify({'error': f'Missing required columns: {missing_columns}'}), 400
            
        # Perform live training
        results = classifier.live_train(new_data)
        
        # Get training history
        history = classifier.get_training_history()
        
        return jsonify({
            'message': 'Training completed successfully',
            'results': results,
            'history': history
        })
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during training: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001)) 
    app.run(debug=True, host='0.0.0.0', port=port)