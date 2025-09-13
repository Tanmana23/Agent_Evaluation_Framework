import pandas as pd
import os
from tqdm import tqdm

# --- IMPORT ALL SCORING MODULES ---

# Import from your existing, completed scorers
from scorers1 import AdvancedInstructionFollowingScorer, score_coherence
from novel_scoring_methods import (
    score_cognitive_agility_v2,
    score_sentiment_risk,
    score_rhetoric_analysis
)

# --- CORRECTED IMPORT SECTION ---
# Import from the correct file name as you specified
try:
    from advanced_scorers import score_contradiction_hallucination, score_assumption
except ImportError:
    print("Warning: Could not import from 'advanced_scores.py'. Make sure the file exists.")
    # Define dummy functions so the script can still run without crashing
    def score_contradiction_hallucination(response, ground_truth): return -1.0
    def score_assumption(response): return -1.0


# --- CONFIGURATION ---
INPUT_CSV_PATH = "data/dataset_new.csv"
OUTPUT_CSV_PATH = "data/results_new2.csv" # New output file for the complete results

def main():
    """
    Main function to run the COMPLETE evaluation pipeline, incorporating all
    basic, novel, and advanced scoring methods.
    """
    # --- DATA LOADING ---
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input file not found at {INPUT_CSV_PATH}")
        return

    print(f"Loading dataset from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Dataset loaded successfully with {len(df)} rows.")

    # --- SCORING ---
    all_scores = []
    
    # Instantiate the class-based scorer once for efficiency
    instruction_scorer = AdvancedInstructionFollowingScorer()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Running Final Evaluation Pipeline"):
        response_text = str(row['response_text']) if pd.notna(row['response_text']) else ""
        ground_truth_text = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ""

        scores = {
            'agent_id': row['agent_id'],
            'agent_persona': row['agent_persona'],
            'prompt_text': row['prompt_text'],
            'response_text': response_text,
            'ground_truth': ground_truth_text
        }
        
        # --- Core Scorers ---
        inst_score, inst_exp = instruction_scorer.score(row['prompt_text'], response_text, row['agent_persona'])
        scores['instruction_following_score'] = inst_score
        scores['instruction_following_explanation'] = inst_exp
        scores['coherence_score'] = score_coherence(row['prompt_text'], response_text)

        # --- Novel Scorers ---
        scores['cognitive_agility_score'] = score_cognitive_agility_v2(response_text, ground_truth_text)
        sentiment_risk = score_sentiment_risk(response_text)
        scores['sentiment_safety_score'] = 1.0 - sentiment_risk
        rhetoric_count = score_rhetoric_analysis(response_text)
        scores['rhetoric_honesty_score'] = max(0.0, 1.0 - (rhetoric_count / 5.0))
        
        # --- NEW: Advanced Hallucination & Assumption Scorers ---
        scores['hallucination_score'] = score_contradiction_hallucination(response_text, ground_truth_text)
        scores['assumption_score'] = score_assumption(response_text)
        
        all_scores.append(scores)

    results_df = pd.DataFrame(all_scores)
    
    print("\nScoring complete!")

    # --- SAVE RESULTS ---
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Comprehensive results saved successfully to {OUTPUT_CSV_PATH}")

    # --- SUMMARY REPORT ---
    print("\n--- Final Comprehensive Evaluation Summary ---")
    
    # Define all the columns for the final report
    score_columns = [
        'instruction_following_score',
        'coherence_score',
        'cognitive_agility_score',
        'sentiment_safety_score',
        'rhetoric_honesty_score',
        'hallucination_score', # New
        'assumption_score'     # New
    ]
    
    persona_avg_scores = results_df.groupby('agent_persona')[score_columns].mean()
    print("\nAverage Scores per Persona:")
    print(persona_avg_scores.round(3))

if __name__ == "__main__":
    main()

