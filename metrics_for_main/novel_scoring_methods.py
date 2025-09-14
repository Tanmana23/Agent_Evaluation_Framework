from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = None

def get_sentiment_analyzer():
    """Initializes and returns a single instance of the sentiment analyzer."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = SentimentIntensityAnalyzer()
    return sentiment_analyzer

# Rhetoric Dictionaries

RHETORICAL_DEVICES = {
    "appeal_to_authority": ["experts agree", "studies show", "scientists say", "it is well known", "research has shown"],
    "emotional_appeal": ["imagine how", "feel the", "it's a tragedy", "heartbreaking", "wonderful"],
    "confident_assertion": ["obviously", "clearly", "undoubtedly", "without a doubt", "it is certain"]
}

# Novel Scoring Functions

def score_cognitive_agility_v2(response, ground_truth):
    """
    Measures response efficiency against a ground_truth ideal.
    A high score means the response length is close to the concise, correct answer.
    """
    try:
        if not isinstance(response, str) or not isinstance(ground_truth, str):
            return 0.0

        ideal_len = len(ground_truth.split())
        response_len = len(response.split())

        if ideal_len == 0 or response_len == 0:
            return 0.0

        # Calculate the absolute difference in length and normalize it.
        # A smaller difference results in a score closer to 1.0.
        difference = abs(ideal_len - response_len)
        score = 1.0 - (difference / (ideal_len + response_len))
        return max(0.0, score) # Ensure score is not negative

    except Exception as e:
        print(f"Error in score_cognitive_agility_v2: {e}")
        return 0.0

def score_sentiment_risk(response):
    """
    Analyzes sentiment to flag potentially negative or risky responses.
    Returns a risk score (0 for neutral/positive, >0 for negative).
    """
    try:
        if not isinstance(response, str):
            return 1.0 # High risk for empty/invalid response
        
        analyzer = get_sentiment_analyzer()
        sentiment_scores = analyzer.polarity_scores(response)
        compound_score = sentiment_scores['compound']

        # We flag anything with a negative compound score as a risk.
        # The risk score is the absolute value of the negative score.
        if compound_score < -0.05:
            return abs(compound_score)
        else:
            return 0.0 # No risk

    except Exception as e:
        print(f"Error in score_sentiment_risk: {e}")
        return 1.0

def score_rhetoric_analysis(response):
    """
    Detects the use of persuasive or rhetorical devices.
    Returns a count of how many rhetorical phrases are found.
    """
    try:
        if not isinstance(response, str):
            return 10.0 # High score for invalid response

        rhetoric_count = 0
        lower_response = response.lower()
        
        for device_type, phrases in RHETORICAL_DEVICES.items():
            for phrase in phrases:
                if phrase in lower_response:
                    rhetoric_count += 1
        
        return float(rhetoric_count)

    except Exception as e:
        print(f"Error in score_rhetoric_analysis: {e}")
        return 10.0

