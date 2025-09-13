import pandas as pd
import os
from tqdm import tqdm

# Import from the correct filenames as you specified
from scorers1 import AdvancedInstructionFollowingScorer, score_coherence
from novel_scoring_methods import (
    score_cognitive_agility_v2,
    score_sentiment_risk,
    score_rhetoric_analysis
)

# --- CONFIGURATION ---
INPUT_CSV_PATH = "data/dataset_new.csv"
OUTPUT_CSV_PATH = "data/results_new1.csv"

def main():
    """
    Main function to run the evaluation pipeline with our completed scorers.
    Loads the dataset, applies instruction following, coherence, and all novel 
    scorers, and saves the comprehensive results.
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
    
    # Instantiate the class-based scorer once
    instruction_scorer = AdvancedInstructionFollowingScorer()

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Running Our Evaluation Pipeline"):
        # Ensure response_text is a string to avoid errors
        response_text = str(row['response_text']) if pd.notna(row['response_text']) else ""

        scores = {
            'agent_id': row['agent_id'],
            'agent_persona': row['agent_persona'],
            'prompt_text': row['prompt_text'],
            'response_text': response_text,
            'ground_truth': row['ground_truth']
        }
        
        # --- Our Completed Core Scorers ---
        inst_score, inst_exp = instruction_scorer.score(row['prompt_text'], response_text, row['agent_persona'])
        scores['instruction_following_score'] = inst_score
        scores['instruction_following_explanation'] = inst_exp
        scores['coherence_score'] = score_coherence(row['prompt_text'], response_text)

        # --- Our Novel "Winning" Scorers (with corrected function names) ---
        scores['cognitive_agility_score'] = score_cognitive_agility_v2(response_text, row['ground_truth'])
        
        # Invert sentiment risk: 0 risk = 1.0 score, max risk (1.0) = 0.0 score
        sentiment_risk = score_sentiment_risk(response_text)
        scores['sentiment_safety_score'] = 1.0 - sentiment_risk

        # Invert rhetoric score: high count = low score. We'll cap at 5 for normalization.
        rhetoric_count = score_rhetoric_analysis(response_text)
        scores['rhetoric_honesty_score'] = max(0.0, 1.0 - (rhetoric_count / 5.0))
        
        all_scores.append(scores)

    results_df = pd.DataFrame(all_scores)
    
    print("\nScoring complete!")

    # --- SAVE RESULTS ---
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Results saved successfully to {OUTPUT_CSV_PATH}")

    # --- SUMMARY REPORT ---
    print("\n--- Our Comprehensive Evaluation Summary ---")
    
    # Define the columns we want to see in the final report
    score_columns = [
        'instruction_following_score',
        'coherence_score',
        'cognitive_agility_score',
        'sentiment_safety_score',
        'rhetoric_honesty_score'
    ]
    
    # Calculate and print average scores per persona for our defined columns
    persona_avg_scores = results_df.groupby('agent_persona')[score_columns].mean()
    print("\nAverage Scores per Persona:")
    print(persona_avg_scores.round(3))

if __name__ == "__main__":
    main()

