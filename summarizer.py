# ----------------------------------------- Cach thu 1: Summarize article thanh 1 headline -------------------------------------

# from transformers import pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ARTICLE = """ Netflix (NFLX) stock climbed in after-hours trading on Thursday after the company delivered first-quarter earnings that beat expectations on both the top and bottom lines and also reiterated full-year revenue guidance.

# Netflix reported revenue of $10.54 billion in the first quarter, a 13% year-over-year jump and a beat compared to Bloomberg analyst expectations of $10.50 billion. The company had guided to $10.42 billion.
# Earnings per share of $6.61 also beat analyst estimates of $5.68. The company had expected earnings of $5.58 in the first quarter after reporting $5.28 in the prior-year period.

# The company guided to revenue for the current quarter above Wall Street expectations, forecasting Q2 revenue of $11.04 billion compared to the $10.88 billion analysts polled by Bloomberg had expected.

# For full-year 2025, the company reiterated its prior forecast of $43.5 billion to $44.5 billion revenue growth and operating margins of 29%.
# The results come as the company currently sits in one of the best positions among Big Tech names amid an uncertain economic environment dominated by President Trump's trade war.

# Netflix stock was up 9.2% this year through Thursday's close, a standout when measured against year-to-date declines of 17% or more for its larger tech peers, including Apple (AAPL), Amazon (AMZN), and Alphabet (GOOG, GOOGL). The S&P 500 (^GSPC) is off about 10% in 2025.
# "We are off to a good start in 2025," Netflix management said in the earnings release, crediting its results to "slightly higher subscription and ad revenue."
# Thursday also marked Netflix's first report without subscriber numbers as the company focuses on driving greater engagement and top-line growth.

# At the end of 2024, the company had 301.6 million global subscribers. Netflix said in its fourth quarter shareholder letter it will disclose subscriber data in the future "as we cross key milestones." The company added 41 million global subscribers last year.

# According to the Wall Street Journal, Netflix is targeting lofty financial goals, which include doubling its revenue by 2030 and reaching a valuation of $1 trillion. The streamer's market cap is currently just north of $400 billion.
# Password-sharing crackdowns helped aid its subscriber figures, and although the benefits of those crackdowns are expected to slow in the near term, the company expects subscriber upside from its content slate, with its ad tier serving as a longer-term catalyst for capturing new users.

# Earlier this year, the company raised subscription prices across its various streaming tiers in the US, including the ad plan, which still remains one of the cheapest tiers on the market at $7.99 per month.

# "The recent pricing adjustments we made in large markets (including US, UK and Argentina) have performed in line with our expectations," Netflix said in the earnings release.

# On Thursday, the company announced it would be raising prices in France, starting today.
# """
# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))


# ----------------------------------------- Cach thu 2: Run article through LLM and generate a list of relevant sentences -------------------------------------
import requests
import json

def summarize(article_text, api_key, model_name="deepseek/deepseek-r1:free"):
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