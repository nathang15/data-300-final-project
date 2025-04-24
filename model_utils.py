import pickle
import os
import json

def save_model(model, model_path='models/sentiment_model.pkl'):
    directory = os.path.dirname(model_path)
    os.makedirs(directory, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(model_path='models/sentiment_model.pkl'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model

def load_articles(file_path='data/examples.json'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Articles file not found at {file_path}")
        
    with open(file_path, 'r') as f:
        articles_dict = json.load(f)
    print(f"Articles loaded from {file_path}")
    return articles_dict

def check_model_exists(model_path='sentiment_model.pkl'):
    return os.path.exists(model_path)

def load_article(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Article file not found at {file_path}")
        
    with open(file_path, 'r') as f:
        article_data = json.load(f)

    return article_data['content']