import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
import os
import requests
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def download_loughran_mcdonald_lexicon(directory=None):
    """
    Downloads and processes the Loughran-McDonald sentiment lexicon.
    """
    if directory is None:
        directory = os.getcwd()
    os.makedirs(directory, exist_ok=True)
    
    # File paths
    raw_file_path = os.path.join(directory, "LoughranMcDonald_MasterDictionary.csv")
    processed_file_path = os.path.join(directory, "loughran_mcdonald_lexicon.csv")

    if os.path.exists(processed_file_path):
        return pd.read_csv(processed_file_path)

    if not os.path.exists(raw_file_path):
        url = "https://drive.google.com/uc?id=12ECPJMxV2wSalXG8ykMmkpa1fq_ur0Rf&export=download"
        print(f"Downloading Loughran-McDonald lexicon to {raw_file_path}...")
        response = requests.get(url)
        with open(raw_file_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    
    print("Processing Loughran-McDonald lexicon...")
    data = pd.read_csv(raw_file_path)
    cols = ['Word', 'Negative', 'Positive', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous']
    df = data[cols]
    words = []
    sentiments = []
    
    # Process each sentiment category
    for sentiment in ['Negative', 'Positive', 'Uncertainty', 'Litigious', 'Constraining', 'Superfluous']:
        category_words = df[df[sentiment] != 0]['Word'].tolist()
        for word in category_words:
            words.append(word.lower())
            sentiments.append(sentiment.lower())

    processed_df = pd.DataFrame({
        'word': words,
        'sentiment': sentiments
    })
    
    processed_df.to_csv(processed_file_path, index=False)
    print(f"Processed lexicon saved to {processed_file_path}")
    
    return processed_df


def load_financial_sentiment_lexicon(directory=None):
    """
    Load the Loughran-McDonald financial sentiment lexicon.
    
    Returns:
    --------
    dict
        Dictionary with sentiment categories as keys and lists of words as values
    """
    lexicon_df = download_loughran_mcdonald_lexicon(directory)
    financial_sentiment_lexicon = {}
    grouped = lexicon_df.groupby('sentiment')['word'].apply(list)
    for sentiment, words in grouped.items():
        financial_sentiment_lexicon[sentiment] = words
    
    print(f"Loaded financial sentiment lexicon with {len(lexicon_df)} words across {len(financial_sentiment_lexicon)} categories")
    
    return financial_sentiment_lexicon

def load_dataset(file_path):
    """Load the Financial Phrasebank dataset."""
    df = pd.read_csv(file_path, 
                    header=None,
                    names=['sentiment', 'text'],
                    encoding='latin1')

    return df

def plot_sentiment_distribution(df):
    """Plot the sentiment distribution."""
    sentiment_counts = df['sentiment'].value_counts().sort_values(ascending=False)
    
    plot_df = pd.DataFrame({'sentiment': sentiment_counts.index, 'count': sentiment_counts.values})
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='sentiment', y='count', data=plot_df, palette='viridis')
    plt.title('Distribution of Sentiment Labels')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.show()
    
    print("\nSentiment distribution:")
    sentiment_percentages = (sentiment_counts / len(df) * 100).round(1)
    for label, percentage in sentiment_percentages.items():
        print(f"{label}: {percentage}%")

def preprocess_text(text, financial_sentiment_lexicon=None):
    """
    Preprocess text for financial sentiment analysis.
    
    Parameters:
    -----------
    text : str
        Text to preprocess
    financial_sentiment_lexicon : dict, optional
        Dictionary with sentiment categories as keys and lists of words as values
        
    Returns:
    --------
    tuple
        (stemmed_tokens, processed_text)
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    tokens = word_tokenize(text)
    stemmed_tokens = []

    preserve_words = set()
    if financial_sentiment_lexicon:
        for words in financial_sentiment_lexicon.values():
            preserve_words.update(words)

    for token in tokens:
        if token not in stop_words and token.strip():
            # append the preserve words in the financial sentiment lexicon as is without stemming to categorize
            # it into either HAS_NEGATIVE, HAS_POSITIVE later
            if token in preserve_words:
                stemmed_tokens.append(token)
            else:
                stemmed = stemmer.stem(token)
                stemmed_tokens.append(stemmed)

    if financial_sentiment_lexicon:
        for sentiment, words in financial_sentiment_lexicon.items():
            if any(word in tokens for word in words):
                stemmed_tokens.append(f"HAS_{sentiment.upper()}")
    
    return stemmed_tokens, ' '.join(stemmed_tokens)

def preprocess_dataset(df, financial_sentiment_lexicon=None):
    print("\nPreprocessing dataset...")
    df['stemmed_tokens'] = None
    df['processed_text'] = None
    
    for idx, row in df.iterrows():
        tokens, text = preprocess_text(row['text'], financial_sentiment_lexicon)
        df.at[idx, 'stemmed_tokens'] = tokens
        df.at[idx, 'processed_text'] = text
    
    return df

def create_stemmed_wordcloud(df):
    """Create a word cloud from stemmed tokens"""
    all_stemmed_words = []
    for tokens in df['stemmed_tokens']:
        all_stemmed_words.extend(tokens)
    
    word_freq = Counter(all_stemmed_words)

    # Create and display wordcloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white', 
                          max_words=100, 
                          contour_width=3).generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Stemmed Financial Texts')
    plt.savefig('stemmed_wordcloud.png')
    plt.show()
    
    return word_freq