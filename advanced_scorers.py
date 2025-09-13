from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import torch

# --- CRITICAL: MODEL CACHING MECHANISM ---
# This global dictionary will store our large, pre-trained models.
# By storing them here, we ensure that each model is loaded from disk
# into memory ONLY ONCE during the entire pipeline run.
MODEL_CACHE = {}

def get_model(model_name, model_class, tokenizer_class=None):
    """
    Robustly loads and caches any transformer model to prevent re-loading.
    If a model is already in MODEL_CACHE, it is returned instantly.
    """
    # Check if the model is already in our cache.
    if model_name in MODEL_CACHE:
        # If yes, retrieve it from memory instead of loading from disk.
        # print(f"Retrieving '{model_name}' from cache...") # Uncomment for debugging
        return MODEL_CACHE[model_name]

    # If the model is not in the cache, load it for the first time.
    print(f"Loading '{model_name}' model for the first time (this may take a moment)...")
    if tokenizer_class:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        # Store the loaded model in the cache for future use.
        MODEL_CACHE[model_name] = {"model": model, "tokenizer": tokenizer}
    else: # For sentence-transformers that don't have a separate tokenizer.
        model = model_class(model_name)
        MODEL_CACHE[model_name] = model
        
    print(f"'{model_name}' loaded and cached successfully.")
    return MODEL_CACHE[model_name]

# --- FINAL SCORING FUNCTIONS ---

def score_contradiction_hallucination(response, ground_truth):
    """
    Uses a Natural Language Inference (NLI) model to detect logical contradictions.
    """
    if not all(isinstance(text, str) and text for text in [response, ground_truth]):
        return 0.5

    try:
        nli_model_name = 'roberta-large-mnli'
        # This will be slow the FIRST time it's called, and instant every time after.
        nli_components = get_model(nli_model_name, AutoModelForSequenceClassification, AutoTokenizer)
        tokenizer = nli_components['tokenizer']
        model = nli_components['model']

        tokenized_input = tokenizer(ground_truth, response, return_tensors='pt', truncation=True, max_length=512)
        logits = model(**tokenized_input).logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]
        
        contradiction_prob = probabilities[model.config.label2id['CONTRADICTION']]
        return 1.0 - contradiction_prob

    except Exception as e:
        print(f"Warning: NLI hallucination check failed: {e}. Skipping.")
        return 0.5

def score_assumption(response):
    """
    HYBRID: Combines a fast regex check with a powerful semantic check.
    """
    if not isinstance(response, str) or not response:
        return 1.0

    # Layer 1: Fast Regex Filter
    assumption_phrases = [
        r"\bi assume\b", r"\bmy assumption is\b", r"\bassuming that\b", r"\bi'm guessing\b",
        r"\byou're probably\b", r"\byou're likely\b", r"\byou likely mean\b", r"\byou must be asking\b",
        r"\bfor your homework\b", r"\bthis probably relates to\b",
        r"\bof course, you know\b", r"\bobviously, you're asking\b"
    ]
    if re.search("|".join(assumption_phrases), response, re.IGNORECASE):
        return 0.0

    # Layer 2: Semantic Deep Dive
    try:
        # This will also use the caching mechanism.
        semantic_model = get_model('all-MiniLM-L6-v2', SentenceTransformer)
        
        assumption_templates = [
            "This is probably for your school report.",
            "My guess is you are a beginner looking for simple answers.",
            "I will assume you are referring to the most recent event.",
            "Since you are asking a basic question, I will simplify this."
        ]

        response_embedding = semantic_model.encode(response, convert_to_tensor=True)
        template_embeddings = semantic_model.encode(assumption_templates, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(response_embedding, template_embeddings)

        if torch.max(cosine_scores) > 0.7:
            return 0.0

    except Exception as e:
        print(f"Warning: Semantic assumption check failed: {e}. Skipping.")
        return 1.0
            
    return 1.0
