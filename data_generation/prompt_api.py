import os
from datasets import load_dataset
import pandas as pd

def prepare_prompts():
    """
    Loads questions from the TruthfulQA dataset and saves them to a CSV file.
    This provides a clean, reusable input for the main data generation script.
    """
    print("Loading TruthfulQA dataset...")
    try:
        truthfulqa = load_dataset("truthful_qa", "generation")
        questions = truthfulqa['validation']['question']
        correct_answers = truthfulqa['validation']['best_answer']
        print(f"Loaded {len(questions)} questions from TruthfulQA.")
        
    except Exception as e:
        print(f" TruthfulQA loading failed: {e}. Using fallback questions.")
        questions = [
            "What is the capital of France?", "How many planets are in our solar system?",
            "Who wrote Romeo and Juliet?", "What is the largest mammal?",
            "When did World War II end?",
        ] * 20
        correct_answers = [
            "Paris", "Eight", "William Shakespeare", "Blue whale", "1945",
        ] * 20

    print(f"Total questions prepared: {len(questions)}")
    
    prompts_df = pd.DataFrame({
        'prompt_text': questions,
        'ground_truth': correct_answers
    })
    return prompts_df

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    prompts_df = prepare_prompts()
    output_path = "../data/prompts_api.csv"
    prompts_df.to_csv(output_path, index=False)
    print(f"\n Prompts successfully saved to {output_path}!")

