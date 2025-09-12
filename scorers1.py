import re
from sentence_transformers import SentenceTransformer, util

# --- MODEL CACHING ---
# Use a dictionary to cache models so they are only loaded once.
models = {}

def get_model(model_name):
    """Efficiently loads and caches a model."""
    if model_name not in models:
        print(f"Loading '{model_name}' model for the first time...")
        models[model_name] = SentenceTransformer(model_name)
        print("Model loaded successfully.")
    return models[model_name]

# --- SCORING FUNCTIONS ---

def score_instruction_following(response, instruction):
    """
    (Placeholder) A simple scorer for instruction following.
    For the TruthfulQA dataset, most instructions are implicit, so we default to a pass.
    """
    # This is a simple placeholder for Day 1. It returns a "pass" score for every response.
    # We will build a more advanced version later if needed.
    return 1.0

def score_coherence(prompt, response):
    """
    Calculates the semantic similarity between the prompt and the response.
    A high score means the response is on-topic.
    """
    try:
        # Ensure response is a string, handle potential float/NaN values from the CSV
        if not isinstance(response, str):
            return 0.0 # Return a failing score for non-string/empty responses

        model = get_model('all-MiniLM-L6-v2')
        embedding1 = model.encode(prompt, convert_to_tensor=True)
        embedding2 = model.encode(response, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item()
    except Exception as e:
        print(f"Error in score_coherence for response '{response[:50]}...': {e}")
        return 0.0 # Return a failing score if embedding fails

