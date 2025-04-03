import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

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

# Text preprocessing and stemming
def create_stemmed_wordcloud(df):
    """Create a word cloud from stemmed text."""
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Process all texts
    all_stemmed_words = []
    
    for text in df['text']:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and stem
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
        all_stemmed_words.extend(stemmed_tokens)
    
    # Count word frequencies
    word_freq = Counter(all_stemmed_words)
    
    print("\nTop 20 most common stemmed words:")
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")
    
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

df = load_dataset('all-data.csv')
plot_sentiment_distribution(df)
create_stemmed_wordcloud(df)