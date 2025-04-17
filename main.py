from load_data import *
from model import *
import time
import os

def main():
    print("Financial Sentiment Analysis with Custom Multinomial Logistic Regression")
    print("=" * 70)
    start_time = time.time()

    os.makedirs('data', exist_ok=True)
    
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

    # Examples
    examples = [
        "The company reported a significant increase in quarterly profits.",
        "Shares plummeted 15% after the disappointing earnings report.",
        "The merger is expected to be completed by the end of the fiscal year.",
        "Analysts have maintained a neutral outlook on the stock.",
        "The company announced layoffs affecting 5% of its workforce.",
        "Revenue grew by 8% year-over-year, meeting market expectations.",
        "The regulatory authority imposed a $2 million fine on the bank.",
        "The startup secured $50 million in Series B funding.",
        "Interest rates remained unchanged following the central bank meeting.",
        "The board of directors approved a dividend of $0.25 per share."
    ]
    
    test_model_on_examples(model, examples, 
                          lambda text: preprocess_text(text, financial_sentiment_lexicon))
    
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()