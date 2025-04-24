from load_data import *
from model import *
import time
import os
from summarizer import *
from model_utils import *

def main():
    API_KEY = "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"
    print("Financial Sentiment Analysis with Custom Multinomial Logistic Regression")
    print("=" * 70)
    start_time = time.time()

    os.makedirs('data', exist_ok=True)
    
    # Load the Loughran-McDonald financial sentiment lexicon
    financial_sentiment_lexicon = load_financial_sentiment_lexicon('data')
    model_path = 'models/sentiment_model.pkl'
    if not check_model_exists(model_path):
        print("No saved model found. Training a new model...")
        
        # Load the Loughran-McDonald financial sentiment lexicon
        financial_sentiment_lexicon = load_financial_sentiment_lexicon('data')

        file_path = 'all-data.csv'
        df = load_dataset(file_path)
        print(f"Loaded dataset with {len(df)} samples")
        plot_sentiment_distribution(df)
        
        # Preprocess the dataset
        df = preprocess_dataset(df, financial_sentiment_lexicon)
        
        # Create word cloud from stemmed tokens
        word_freq = create_stemmed_wordcloud(df)
        
        # Build and evaluate model
        model, accuracy = build_and_evaluate_model(df)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        
        # Save the trained model
        save_model(model, model_path)
    else:
        print(f"Loading saved model from {model_path}...")
        model = load_model(model_path)
        financial_sentiment_lexicon = load_financial_sentiment_lexicon('data')
    
    article_path = 'example_articles/apple.json'
    
    try:
        print(f"\nLoading article from {article_path}...")
        article_text = load_article(article_path)
        
        print("\nAnalyzing sentiment...")
        sentences = summarize(article_text, API_KEY)
        
        result = run(model, sentences, lambda text: preprocess_text(text, financial_sentiment_lexicon))
        
        print("\nOverall sentiment:", result)
        
    except FileNotFoundError:
        print(f"Error: Article file not found at {article_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    result = run(model, sentences, lambda text: preprocess_text(text, financial_sentiment_lexicon))
    
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    print("Overall sentiment: " + result)
if __name__ == "__main__":
    main()