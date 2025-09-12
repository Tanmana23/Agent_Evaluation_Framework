import pandas as pd
from tqdm import tqdm
from scorers import score_instruction_following, score_coherence
import os

# --- CONFIGURATION ---
INPUT_DATASET_PATH = "data/dataset.csv"
OUTPUT_RESULTS_PATH = "data/results.csv"

def run_evaluation_pipeline():
    """
    Loads the generated dataset, applies all scoring functions, and saves the
    results to a new CSV file.
    """
    # Check if the input file exists
    if not os.path.exists(INPUT_DATASET_PATH):
        print(f"Error: Input file not found at '{INPUT_DATASET_PATH}'")
        print("Please run 'python prepare_prompts.py' and 'python generate_dataset.py' first.")
        return

    print(f"Loading dataset from {INPUT_DATASET_PATH}...")
    df = pd.read_csv(INPUT_DATASET_PATH)
    print(f"Dataset loaded with {len(df)} responses.")

    # Use tqdm for a progress bar, making it easy to track progress on large datasets.
    tqdm.pandas(desc="Applying Scorers")

    # Apply the scoring functions to each row of the DataFrame.
    # 'progress_apply' is a tqdm feature that adds a progress bar to the pandas apply function.
    print("Scoring for Instruction Following...")
    df['instruction_following_score'] = df.progress_apply(
        lambda row: score_instruction_following(row['prompt_text'], row['response_text']), axis=1
    )

    print("\nScoring for Coherence...")
    df['coherence_score'] = df.progress_apply(
        lambda row: score_coherence(row['prompt_text'], row['response_text']), axis=1
    )

    print("\nScoring complete!")
    
    # Save the enriched DataFrame to a new CSV file.
    df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    print(f"Results saved successfully to {OUTPUT_RESULTS_PATH}")

    # Display a summary of the results
    print("\n--- Evaluation Summary ---")
    print(df.head())
    
    # Calculate and display the average scores for each agent persona
    persona_scores = df.groupby('agent_persona')[['instruction_following_score', 'coherence_score']].mean()
    print("\nAverage Scores per Persona:")
    print(persona_scores.round(3))
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    run_evaluation_pipeline()

