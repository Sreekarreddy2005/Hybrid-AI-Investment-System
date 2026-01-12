from langchain_community.tools.tavily_search import TavilySearchResults

def get_sentiment(ticker: str) -> dict:
    """Fetch recent news headlines and derive sentiment."""
    try:
        search = TavilySearchResults(max_results=10)
        news = search.invoke(f"{ticker} stock latest news headlines sentiment")
        
        positive_keywords = ["growth", "beats", "strong", "profit", "record", "expansion", "upgrade", "bullish"]
        negative_keywords = ["decline", "miss", "weak", "loss", "downgrade", "lawsuit", "risk", "bearish", "crash"]
        
        score = 0
        for item in news:
            text = (item.get("content") or "").lower()
            for word in positive_keywords:
                if word in text: score += 1
            for word in negative_keywords:
                if word in text: score -= 1
        
        if score > 0:
            sentiment = "Positive"
        elif score < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "sentiment": sentiment,
            "score": score,
            "newscount": len(news)
        }
    except Exception as e:
        return {
            "sentiment": "Neutral",
            "score": 0,
            "newscount": 0,
            "error": str(e)
        }
