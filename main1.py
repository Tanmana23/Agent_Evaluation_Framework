import pandas as pd
from tqdm import tqdm
import os
import scorers # Import our Day 1 scoring module

def run_evaluation_pipeline():
    """
    Main function for Day 1.
    It loads the generated dataset and applies the basic scoring functions.
    """
    # --- 1. Load Dataset ---
    dataset_path = 'data/dataset_new.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at '{dataset_path}'.")
        print("Please run a data generation script first.")
        return

    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with {len(df)} responses.")

    # --- 2. Initialize Results Lists ---
    instruction_scores = []
    coherence_scores = []
    
    # --- 3. Run Scoring Loop ---
    # Use tqdm to show a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Applying Day 1 scores"):
        # A. Instruction Following Score
        instruction_score = scorers.score_instruction_following(row['response_text'], "") # Instruction is blank for now
        instruction_scores.append(instruction_score)
        
        # B. Coherence Score
        coherence_score = scorers.score_coherence(row['prompt_text'], row['response_text'])
        coherence_scores.append(coherence_score)

    # --- 4. Add Scores to DataFrame and Save ---
    df['instruction_following_score'] = instruction_scores
    df['coherence_score'] = coherence_scores
    
    results_path = 'data/results_new.csv'
    df.to_csv(results_path, index=False)
    print(f"\nScoring complete!")
    print(f"Results saved successfully to {results_path}")

    # --- 5. Print Summary Report ---
    print("\n--- Day 1 Evaluation Summary ---")
    
    summary = df.groupby('agent_persona')[['instruction_following_score', 'coherence_score']].mean()
    print("Average Scores per Persona:")
    print(summary.round(3))

if __name__ == "__main__":
    run_evaluation_pipeline()

