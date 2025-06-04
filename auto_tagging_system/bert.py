import torch
from transformers import pipeline
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DualModelSentimentAnalyzer:
    def __init__(self):
        """Initialize the dual-model sentiment analyzer"""
        print("Initializing Dual-Model Sentiment Analyzer...")
        print("-" * 50)
        
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Load Model 1: DistilBERT
        print("\nLoading Model 1 (DistilBERT)...")
        try:
            self.model1 = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device
            )
            print("✓ Model 1 loaded successfully!")
        except Exception as e:
            print(f"Error loading Model 1: {e}")
            print("Using default sentiment model...")
            self.model1 = pipeline("sentiment-analysis", device=self.device)
        
        # Load Model 2: Alternative model
        print("\nLoading Model 2 (BERT-based)...")
        try:
            # Try RoBERTa first
            self.model2 = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=self.device
            )
            print("✓ Model 2 (RoBERTa) loaded successfully!")
        except:
            try:
                # Fallback to BERTweet
                self.model2 = pipeline(
                    "sentiment-analysis", 
                    model="finiteautomata/bertweet-base-sentiment-analysis",
                    device=self.device
                )
                print("✓ Model 2 (BERTweet) loaded successfully!")
            except:
                # Final fallback
                print("Using alternative model for Model 2...")
                self.model2 = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=self.device
                )
        
        print("\n✓ Both models loaded and ready!")
        print("-" * 50)
    
    def convert_to_score(self, result):
        """Convert sentiment labels to numerical scores (-1 to 1)"""
        label = result['label'].lower()
        score = result['score']
        
        # Handle different label formats
        if 'negative' in label or 'neg' in label or label == 'label_0':
            return -score
        elif 'positive' in label or 'pos' in label or label == 'label_2':
            return score
        elif 'neutral' in label or 'neu' in label or label == 'label_1':
            return 0
        # Handle star ratings (1-5 stars)
        elif 'star' in label:
            try:
                star_num = int(label.split()[0])
                # Convert 1-5 scale to -1 to 1 scale
                return (star_num - 3) / 2
            except:
                return 0
        else:
            # Default: positive if score > 0.5, else negative
            return score if score > 0.5 else -score
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of a single text using both models"""
        # Truncate text if too long (max 512 tokens)
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        # Get predictions from both models
        result1 = self.model1(text)[0]
        result2 = self.model2(text)[0]
        
        # Convert to numerical scores
        score1 = self.convert_to_score(result1)
        score2 = self.convert_to_score(result2)
        
        # Calculate mean score
        mean_score = (score1 + score2) / 2
        
        # Determine sentiment category
        if mean_score < -0.3:
            sentiment = "NEGATIVE"
            emoji = "😞"
        elif mean_score > 0.3:
            sentiment = "POSITIVE"
            emoji = "😊"
        else:
            sentiment = "NEUTRAL"
            emoji = "😐"
        
        return {
            'text': text[:100] + "..." if len(text) > 100 else text,
            'model1_result': result1,
            'model2_result': result2,
            'model1_score': score1,
            'model2_score': score2,
            'mean_score': mean_score,
            'sentiment': sentiment,
            'emoji': emoji
        }
    
    def analyze_multiple(self, texts):
        """Analyze multiple texts and return results"""
        results = []
        print(f"\nAnalyzing {len(texts)} texts...")
        print("-" * 80)
        
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}] Analyzing: \"{text[:50]}...\"" if len(text) > 50 else f"\n[{i}] Analyzing: \"{text}\"")
            result = self.analyze_sentiment(text)
            results.append(result)
            
            # Print results
            print(f"   Model 1: {result['model1_result']['label']} (confidence: {result['model1_result']['score']:.3f})")
            print(f"   Model 2: {result['model2_result']['label']} (confidence: {result['model2_result']['score']:.3f})")
            print(f"   → Mean Score: {result['mean_score']:.3f}")
            print(f"   → Final Sentiment: {result['sentiment']} {result['emoji']}")
        
        return results
    
    def visualize_results(self, results):
        """Create visualization of sentiment analysis results"""
        if not results:
            print("No results to visualize!")
            return
        
        # Prepare data
        texts = [r['text'][:30] + "..." if len(r['text']) > 30 else r['text'] for r in results]
        model1_scores = [r['model1_score'] for r in results]
        model2_scores = [r['model2_score'] for r in results]
        mean_scores = [r['mean_score'] for r in results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Bar chart of scores
        x = np.arange(len(texts))
        width = 0.25
        
        ax1.bar(x - width, model1_scores, width, label='Model 1', alpha=0.8)
        ax1.bar(x, model2_scores, width, label='Model 2', alpha=0.8)
        ax1.bar(x + width, mean_scores, width, label='Mean Score', alpha=0.8, color='green')
        
        ax1.set_ylabel('Sentiment Score')
        ax1.set_title('Sentiment Analysis Results - Model Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(texts, rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=0.5, label='Positive threshold')
        ax1.axhline(y=-0.3, color='red', linestyle='--', linewidth=0.5, label='Negative threshold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sentiment distribution
        sentiments = [r['sentiment'] for r in results]
        sentiment_counts = {
            'POSITIVE': sentiments.count('POSITIVE'),
            'NEUTRAL': sentiments.count('NEUTRAL'),
            'NEGATIVE': sentiments.count('NEGATIVE')
        }
        
        colors = ['green', 'gray', 'red']
        ax2.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=colors)
        ax2.set_title('Overall Sentiment Distribution')
        
        plt.tight_layout()
        plt.show()


# Interactive usage example
def interactive_analysis():
    """Run interactive sentiment analysis"""
    analyzer = DualModelSentimentAnalyzer()
    
    print("\n" + "="*50)
    print("INTERACTIVE SENTIMENT ANALYSIS")
    print("="*50)
    print("Enter 'quit' to exit, 'batch' for batch analysis, or type any text to analyze")
    
    while True:
        print("\n" + "-"*50)
        user_input = input("\nEnter text to analyze: ").strip()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the sentiment analyzer!")
            break
        
        elif user_input.lower() == 'batch':
            print("\nBATCH ANALYSIS MODE")
            print("Enter texts one per line. Enter 'done' when finished:")
            texts = []
            while True:
                text = input(f"Text {len(texts)+1}: ").strip()
                if text.lower() == 'done':
                    break
                if text:
                    texts.append(text)
            
            if texts:
                results = analyzer.analyze_multiple(texts)
                
                # Show summary
                print("\n" + "="*50)
                print("SUMMARY")
                print("="*50)
                mean_overall = np.mean([r['mean_score'] for r in results])
                print(f"Overall average sentiment: {mean_overall:.3f}")
                print(f"Most positive: {max(results, key=lambda x: x['mean_score'])['text']}")
                print(f"Most negative: {min(results, key=lambda x: x['mean_score'])['text']}")
                
                # Visualize
                visualize = input("\nVisualize results? (y/n): ").lower()
                if visualize == 'y':
                    analyzer.visualize_results(results)
        
        elif user_input:
            # Single text analysis
            result = analyzer.analyze_sentiment(user_input)
            
            print("\n" + "="*50)
            print("SENTIMENT ANALYSIS RESULT")
            print("="*50)
            print(f"\nText: \"{result['text']}\"")
            print(f"\nModel 1 ({result['model1_result']['label']}): Score = {result['model1_score']:.3f}")
            print(f"Model 2 ({result['model2_result']['label']}): Score = {result['model2_score']:.3f}")
            print(f"\nMean Score: {result['mean_score']:.3f}")
            print(f"Final Sentiment: {result['sentiment']} {result['emoji']}")
            
            # Interpretation
            print("\nInterpretation:")
            if result['mean_score'] < -0.7:
                print("→ Very negative sentiment detected")
            elif result['mean_score'] < -0.3:
                print("→ Negative sentiment detected")
            elif result['mean_score'] < 0.3:
                print("→ Neutral sentiment detected")
            elif result['mean_score'] < 0.7:
                print("→ Positive sentiment detected")
            else:
                print("→ Very positive sentiment detected")


# Example usage with predefined sentences
def demo_analysis():
    """Demo the analyzer with example sentences"""
    analyzer = DualModelSentimentAnalyzer()
    
    # Example sentences
    demo_sentences = [
        "I absolutely love this product! It exceeded all my expectations.",
        "The service was terrible and the staff was very rude.",
        "It's okay, nothing special but not bad either.",
        "This is the worst experience I've ever had with a bank!",
        "Thank you so much for your excellent customer service!",
        "The product works as described.",
        "I'm extremely disappointed with the quality.",
        "Best purchase I've made this year!",
    ]
    
    print("\n" + "="*50)
    print("DEMO ANALYSIS")
    print("="*50)
    
    results = analyzer.analyze_multiple(demo_sentences)
    
    # Visualize results
    analyzer.visualize_results(results)
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    scores = [r['mean_score'] for r in results]
    print(f"Average sentiment score: {np.mean(scores):.3f}")
    print(f"Standard deviation: {np.std(scores):.3f}")
    print(f"Most positive score: {max(scores):.3f}")
    print(f"Most negative score: {min(scores):.3f}")


if __name__ == "__main__":
    # Choose mode
    print("DUAL-MODEL SENTIMENT ANALYZER")
    print("1. Interactive mode (analyze custom text)")
    print("2. Demo mode (see examples)")
    
    choice = input("\nSelect mode (1 or 2): ").strip()
    
    if choice == "1":
        interactive_analysis()
    elif choice == "2":
        demo_analysis()
    else:
        print("Invalid choice. Running demo mode...")
        demo_analysis()