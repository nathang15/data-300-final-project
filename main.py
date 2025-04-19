from load_data import *
from model import *
import time
import os
from summarizer import *

def main():
    API_KEY = 
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
    
    article = """
    Netflix (NFLX) stock climbed in after-hours trading on Thursday after the company delivered first-quarter earnings that beat expectations on both the top and bottom lines and also reiterated full-year revenue guidance.
    
    Netflix reported revenue of $10.54 billion in the first quarter, a 13% year-over-year jump and a beat compared to Bloomberg analyst expectations of $10.50 billion. The company had guided to $10.42 billion.
    
    Earnings per share of $6.61 also beat analyst estimates of $5.68. The company had expected earnings of $5.58 in the first quarter after reporting $5.28 in the prior-year period.
    
    The company guided to revenue for the current quarter above Wall Street expectations, forecasting Q2 revenue of $11.04 billion compared to the $10.88 billion analysts polled by Bloomberg had expected.
    
    For full-year 2025, the company reiterated its prior forecast of $43.5 billion to $44.5 billion revenue growth and operating margins of 29%.
    
    The results come as the company currently sits in one of the best positions among Big Tech names amid an uncertain economic environment dominated by President Trump's trade war.
    
    Netflix stock was up 9.2% this year through Thursday's close, a standout when measured against year-to-date declines of 17% or more for its larger tech peers, including Apple (AAPL), Amazon (AMZN), and Alphabet (GOOG, GOOGL). The S&P 500 (^GSPC) is off about 10% in 2025.
    
    "We are off to a good start in 2025," Netflix management said in the earnings release, crediting its results to "slightly higher subscription and ad revenue."
    
    Thursday also marked Netflix's first report without subscriber numbers as the company focuses on driving greater engagement and top-line growth.
    
    At the end of 2024, the company had 301.6 million global subscribers. Netflix said in its fourth quarter shareholder letter it will disclose subscriber data in the future "as we cross key milestones." The company added 41 million global subscribers last year.
    
    According to the Wall Street Journal, Netflix is targeting lofty financial goals, which include doubling its revenue by 2030 and reaching a valuation of $1 trillion. The streamer's market cap is currently just north of $400 billion.
    
    Password-sharing crackdowns helped aid its subscriber figures, and although the benefits of those crackdowns are expected to slow in the near term, the company expects subscriber upside from its content slate, with its ad tier serving as a longer-term catalyst for capturing new users.
    """
    article2 = """
    Health insurance company UnitedHealth (NYSE:UNH) fell short of the market’s revenue expectations in Q1 CY2025, but sales rose 9.8% year on year to $109.6 billion. Its non-GAAP profit of $7.20 per share was 1.3% below analysts’ consensus estimates.
    UnitedHealth (UNH) Q1 CY2025 Highlights:
    Revenue: $109.6 billion vs analyst estimates of $111.5 billion (9.8% year-on-year growth, 1.7% miss)

    Adjusted EPS: $7.20 vs analyst expectations of $7.29 (1.3% miss)

    Adjusted EBITDA: $10.56 billion vs analyst estimates of $10.55 billion (9.6% margin, in line)

    Adjusted EPS guidance for the full year is $26.25 at the midpoint, missing analyst estimates by 11.7%

    Operating Margin: 8.3%, in line with the same quarter last year

    Free Cash Flow Margin: 4.2%, up from 0.4% in the same quarter last year

    Market Capitalization: $535.1 billion
    “UnitedHealth Group grew to serve more people more comprehensively but did not perform up to our expectations, and we are aggressively addressing those challenges to position us well for the years ahead, and return to our long-term earnings growth rate target of 13 to 16%,” said Andrew Witty, chief executive officer of UnitedHealth Group.

    Company Overview
    With over 100 million people served across its various businesses and a workforce of more than 400,000, UnitedHealth Group (NYSE:UNH) operates a health insurance business and Optum, a healthcare services division that provides everything from pharmacy benefits to primary care.

    Health Insurance Providers
    Upfront premiums collected by health insurers lead to reliable revenue, but profitability ultimately depends on accurate risk assessments and the ability to control medical costs. Health insurers are also highly sensitive to regulatory changes and economic conditions such as unemployment. Going forward, the industry faces tailwinds from an aging population, increasing demand for personalized healthcare services, and advancements in data analytics to improve cost management. However, continued regulatory scrutiny on pricing practices, the potential for government-led reforms such as expanded public healthcare options, and inflation in medical costs could add volatility to margins. One big debate among investors is the long-term impact of AI and whether it will help underwriting, fraud detection, and claims processing or whether it may wade into ethical grey areas like reinforcing biases and widening disparities in medical care.

    Sales Growth
    Examining a company’s long-term performance can provide clues about its quality. Even a bad business can shine for one or two quarters, but a top-tier one grows for years. Over the last five years, UnitedHealth grew its sales at a decent 10.7% compounded annual growth rate. Its growth was slightly above the average healthcare company and shows its offerings resonate with customers.
    We at StockStory place the most emphasis on long-term growth, but within healthcare, a half-decade historical view may miss recent innovations or disruptive industry trends. UnitedHealth’s annualized revenue growth of 10.5% over the last two years aligns with its five-year trend, suggesting its demand was stable.
    This quarter, UnitedHealth’s revenue grew by 9.8% year on year to $109.6 billion, missing Wall Street’s estimates.

    Looking ahead, sell-side analysts expect revenue to grow 12.7% over the next 12 months, an improvement versus the last two years. This projection is particularly noteworthy for a company of its scale and suggests its newer products and services will catalyze better top-line performance.

    Here at StockStory, we certainly understand the potential of thematic investing. Diverse winners from Microsoft (MSFT) to Alphabet (GOOG), Coca-Cola (KO) to Monster Beverage (MNST) could all have been identified as promising growth stories with a megatrend driving the growth. So, in that spirit, we’ve identified a relatively under-the-radar profitable growth stock benefiting from the rise of AI, available to you FREE via this link.

    Operating Margin
    UnitedHealth was profitable over the last five years but held back by its large cost base. Its average operating margin of 8.5% was weak for a healthcare business.

    Analyzing the trend in its profitability, UnitedHealth’s operating margin decreased by 1 percentage points over the last five years. A silver lining is that on a two-year basis, its margin has stabilized. We like UnitedHealth and hope it can right the ship.

    In Q1, UnitedHealth generated an operating profit margin of 8.3%, in line with the same quarter last year. This indicates the company’s overall cost structure has been relatively stable.

    Earnings Per Share
    We track the long-term change in earnings per share (EPS) for the same reason as long-term revenue growth. Compared to revenue, however, EPS highlights whether a company’s growth is profitable.

    UnitedHealth’s EPS grew at a spectacular 13.1% compounded annual growth rate over the last five years, higher than its 10.7% annualized revenue growth. However, this alone doesn’t tell us much about its business quality because its operating margin didn’t expand.

    We can take a deeper look into UnitedHealth’s earnings to better understand the drivers of its performance. A five-year view shows that UnitedHealth has repurchased its stock, shrinking its share count by 4.6%. This tells us its EPS outperformed its revenue not because of increased operational efficiency but financial engineering, as buybacks boost per share earnings.
    In Q1, UnitedHealth reported EPS at $7.20, up from $6.91 in the same quarter last year. Despite growing year on year, this print slightly missed analysts’ estimates, but we care more about long-term EPS growth than short-term movements. Over the next 12 months, Wall Street expects UnitedHealth’s full-year EPS of $27.96 to grow 10.1%.

    Key Takeaways from UnitedHealth’s Q1 Results
    We struggled to find many positives in these results. Its revenue and EPS in the quarter both fell short of Wall Street’s estimates. Looking ahead, full-year EPS guidance missed significantly as well. Overall, this was a softer quarter. The stock traded down 9.5% to $529.60 immediately after reporting.

    The latest quarter from UnitedHealth’s wasn’t that good. One earnings report doesn’t define a company’s quality, though, so let’s explore whether the stock is a buy at the current price. We think that the latest quarter is just one piece of the longer-term business quality puzzle. Quality, when combined with valuation, can help determine if the stock is a buy.
    """
    sentences = summarize(article2, API_KEY)
    
    result = run(model, sentences, lambda text: preprocess_text(text, financial_sentiment_lexicon))
    
    execution_time = time.time() - start_time
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    print("Overall sentiment: " + result)
if __name__ == "__main__":
    main()