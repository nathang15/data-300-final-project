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
        "Netflix (NFLX) stock rose in after-hours trading on Thursday after beating first-quarter earnings expectations on both revenue and earnings per share.",
        "Q1 revenue reached $10.54 billion, up 13% year-over-year, surpassing Bloomberg's estimate of $10.50 billion and Netflix's own guidance of $10.42 billion.",
        "Earnings per share came in at $6.61, beating analyst expectations of $5.68 and the company’s own forecast of $5.58.",
        "Netflix guided Q2 revenue at $11.04 billion, exceeding analyst expectations of $10.88 billion.",
        "For full-year 2025, Netflix reiterated its revenue forecast of $43.5 billion to $44.5 billion and projected operating margins of 29%.",
        "The company highlighted strong performance due to slightly higher subscription and ad revenue.",
        "Netflix stock was up 9.2% year-to-date through Thursday, outperforming larger tech peers like Apple, Amazon, and Alphabet, which saw declines of 17% or more; the S&P 500 was down about 10% in 2025.",
        "This was Netflix’s first earnings report without disclosing subscriber numbers, shifting focus to engagement and revenue growth.",
        "As of the end of 2024, Netflix had 301.6 million global subscribers and added 41 million subscribers during the year.",
        "Netflix aims to double revenue by 2030 and reach a $1 trillion valuation; its current market cap is just over $400 billion.",
        "The crackdown on password sharing has driven recent subscriber growth, though its impact is expected to slow.",
        "Future growth is expected from new content and the ad-supported tier, priced at $7.99/month in the U.S.",
        "Earlier in 2025, Netflix raised prices across U.S. streaming tiers, including the ad plan, and noted that these adjustments performed as expected.",
        "On Thursday, Netflix announced price increases in France, effective immediately."
    ]
    
    test_model_on_examples(model, examples, 
                          lambda text: preprocess_text(text, financial_sentiment_lexicon))
    
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()