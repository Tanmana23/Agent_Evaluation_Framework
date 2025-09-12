import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- State for ML Model ---
# We load the model only once and reuse it to save time and memory.
# It's set to None initially and loaded when the first scoring function needs it.
model = None

def load_model():
    """Helper function to load the SentenceTransformer model on demand."""
    global model
    if model is None:
        print("Loading sentence-transformer model for the first time...")
        # This model is great for comparing the semantic meaning of sentences.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")

# --- SCORING FUNCTIONS ---

def score_instruction_following(prompt, response):
    """
    Placeholder for instruction following. The TruthfulQA dataset doesn't have
    explicit instructions, so we will implement this properly on Day 2 with
    more complex prompts if needed.
    """
    # Returns a neutral score for now.
    return 1.0

def score_coherence(prompt, response):
    """
    Measures how semantically related the response is to the prompt.
    A high score means the response is on-topic.
    """
    # Ensure the ML model is loaded before we use it.
    load_model()
    
    # It's possible for a response to be empty or not a string. We handle this gracefully.
    if not isinstance(response, str) or not response:
        return 0.0

    try:
        # Convert the prompt and response into numerical vectors (embeddings).
        prompt_embedding = model.encode(prompt, convert_to_tensor=True)
        response_embedding = model.encode(response, convert_to_tensor=True)

        # Calculate the cosine similarity between the two vectors.
        # This gives a score from -1 to 1. We scale it to 0-1 for consistency.
        cosine_score = util.pytorch_cos_sim(prompt_embedding, response_embedding).item()
        
        # Scale the score to be between 0 and 1.
        return (cosine_score + 1) / 2
    except Exception as e:
        print(f"Error calculating coherence for response: '{response[:50]}...'. Error: {e}")
        return 0.0

