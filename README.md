# Financial Sentiment Analysis

![Wordcloud of Financial Terms](stemmed_wordcloud.png)

## Overview

This project implements a financial sentiment analysis system using a Multinomial Logistic Regression model. The system analyzes financial news articles to determine sentiment (positive, negative, or neutral) by leveraging domain-specific financial lexicons and advanced text preprocessing techniques.

## Features
- Custom implementation of Multinomial Logistic Regression with:
  - Mini-batch gradient descent
  - L2 regularization
  - Gradient clipping
- Financial-specific text preprocessing:
  - Integration of Loughran-McDonald financial sentiment lexicon
  - Specialized tokenization and stemming
  - Financial feature extraction
- Sentence extraction from financial articles
- Complete sentiment classification pipeline

## Dataset

The project uses the Financial PhraseBank dataset, which contains sentences from financial news articles manually labeled as positive, negative, or neutral. The dataset exhibits the following distribution:

- Neutral: 59.4% (2879 samples)
- Positive: 28.1% (1363 samples)
- Negative: 12.5% (604 samples)

This imbalance reflects the nature of financial reporting, which tends toward neutral, factual communication.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud
- Requests
- tqdm

## Installation

1. Clone this repository:
```
git clone https://github.com/nathang15/data-300-final-project
cd data-300-final-project
pip install -r requirements.txt
```

Set up your OpenRouter API key: https://openrouter.ai/

## Use the project
Simply run main.py file and this will take care of the rest.

Make sure to place the article that you want to analyze in JSON format in the example_articles/ directory
Update the article_path in main.py to point to your article

Example Results
For each analyzed sentence, the model outputs:
- Predicted sentiment (positive, negative, or neutral)
- Confidence score
- Original sentence

Example output:
```
Model predictions:
No.   Prediction      Confidence      Sentence
---------------------------------------------------------------------------
1     positive        0.7842          The company reported revenue of $1.2 billion, up 15% year-over-year.
2     neutral         0.6523          The earnings call is scheduled for next Thursday.
3     negative        0.8102          The company missed analyst expectations by $0.03 per share.
...
```

## Performance
The model achieves:
- Overall accuracy: 65.26%
- Class-specific performance:

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.45 | 0.62 | 0.52 | 121 |
| Neutral | 0.76 | 0.74 | 0.75 | 576 |
| Positive | 0.55 | 0.48 | 0.51 | 273 |
| Macro avg | 0.59 | 0.61 | 0.59 | 970 |
| Weighted avg | 0.66 | 0.65 | 0.65 | 970 |

## License
MIT

## Acknowledgments
Loughran-McDonald Financial Sentiment Lexicon
Financial PhraseBank dataset creators
