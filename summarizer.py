import requests
import json

def summarize(article_text, api_key, model_name="deepseek/deepseek-chat-v3-0324:free"):
    base_url = "https://openrouter.ai/api/v1/chat/completions"
    
    prompt = f"""
    Extract important financial sentences from the following article. Focus on sentences containing:
    - Revenue, profit, earnings, or other financial metrics
    - Growth or percentage changes
    - Stock performance
    - Financial forecasts or guidance
    - Analyst expectations
    
    Article:
    {article_text}
    
    Important financial sentences (return only the sentences, one per line):
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        print("Sending request to OpenRouter API...")
        response = requests.post(
            url=base_url,
            headers=headers,
            data=json.dumps(data)
        )
        
        response.raise_for_status()
        
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            response_text = response_data["choices"][0]["message"]["content"]
            
            sentences = [line.strip() for line in response_text.split('\n') if line.strip()]
            return sentences
        else:
            print("Error: Unexpected response format")
            return []
            
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []