import pandas as pd
import os
import time
from tqdm import tqdm
import numpy as np

# --- IMPORT ALL SCORING MODULES ---
from scorers1 import AdvancedInstructionFollowingScorer, score_coherence
from novel_scoring_methods import score_cognitive_agility_v2, score_sentiment_risk, score_rhetoric_analysis
from advanced_scorers import score_contradiction_hallucination, score_assumption
from ai_judge import AIJudge # Our new module!

# --- CONFIGURATION ---
INPUT_CSV_PATH = "data/dataset_new.csv"
OUTPUT_CSV_PATH = "data/final_results_with_ai_judge.csv"
GEMINI_API_KEY = " " 

# Rate limit handling for AI Judge
REQUESTS_PER_MINUTE = 15
COOLDOWN_SECONDS = 61

def calculate_overall_score(row):
    """
    Calibrates all scores to a 0-100 scale and calculates the final weighted average.
    """
    weights = {
        'hallucination': 0.35,
        'instruction_following': 0.25,
        'assumption': 0.15,
        'coherence': 0.10,
        'novelty_avg': 0.15
    }

    # --- CALIBRATION ---
    # Calibrate NLI hallucination score (0-1 -> 0-100)
    calibrated_hallucination = row['hallucination_score'] * 100

    # If an AI Judge score exists, it's the more expert opinion. We use it to override.
    if pd.notna(row['ai_judge_score']):
        # Calibrate the AI Judge's "harsh grader" score
        raw_judge_score = row['ai_judge_score']
        if raw_judge_score > 0.8: calibrated_hallucination = 100
        elif raw_judge_score > 0.5: calibrated_hallucination = 60 + ((raw_judge_score - 0.5) * (100/3)) # Maps 0.5-0.8 to 60-90
        else: calibrated_hallucination = raw_judge_score * 120 # Maps 0.0-0.5 to 0-60
    
    calibrated_instruction = row['instruction_following_score'] * 100
    calibrated_assumption = row['assumption_score'] * 100
    calibrated_coherence = row['coherence_score'] * 100

    # Calibrate and average the novel scores
    calibrated_agility = row['cognitive_agility_score'] * 100
    calibrated_sentiment = (1.0 - row['sentiment_risk_score']) * 100
    calibrated_rhetoric = max(0, 100 - (row['rhetoric_analysis_score'] * 20))
    novelty_avg = np.mean([calibrated_agility, calibrated_sentiment, calibrated_rhetoric])

    # --- WEIGHTED AVERAGE ---
    overall_score = (
        calibrated_hallucination * weights['hallucination'] +
        calibrated_instruction * weights['instruction_following'] +
        calibrated_assumption * weights['assumption'] +
        calibrated_coherence * weights['coherence'] +
        novelty_avg * weights['novelty_avg']
    )
    return overall_score


def main():
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"Dataset loaded with {len(df)} rows.")

    # --- TIER 1: FAST AUTOMATED SCORING ---
    print("\n--- Running Tier 1: High-Speed Automated Scoring ---")
    instruction_scorer = AdvancedInstructionFollowingScorer()
    
    df['instruction_following_score'], df['instruction_following_explanation'] = zip(*df.apply(lambda row: instruction_scorer.score(str(row['prompt_text']), str(row['response_text']), str(row['agent_persona'])), axis=1))
    df['coherence_score'] = df.apply(lambda row: score_coherence(str(row['prompt_text']), str(row['response_text'])), axis=1)
    df['cognitive_agility_score'] = df.apply(lambda row: score_cognitive_agility_v2(str(row['response_text']), str(row['ground_truth'])), axis=1)
    df['sentiment_risk_score'] = df.apply(lambda row: score_sentiment_risk(str(row['response_text'])), axis=1)
    df['rhetoric_analysis_score'] = df.apply(lambda row: score_rhetoric_analysis(str(row['response_text'])), axis=1)
    df['hallucination_score'] = df.apply(lambda row: score_contradiction_hallucination(str(row['response_text']), str(row['ground_truth'])), axis=1)
    df['assumption_score'] = df.apply(lambda row: score_assumption(str(row['response_text'])), axis=1)
    
    print("Tier 1 scoring complete.")

    # --- TIER 2: INTELLIGENT ESCALATION TO AI JUDGE ---
    print("\n--- Running Tier 2: Intelligent Escalation to AI Judge ---")
    
    suspicion_trigger = (
        (df['hallucination_score'] < 0.75) | 
        (df['rhetoric_analysis_score'] >= 2) |
        (df['assumption_score'] < 0.6) |
        (df['instruction_following_score'] < 0.5)|
        (df['coherence_score'] < 0.35)
    )
    responses_to_judge = df[suspicion_trigger]
    print(f"Identified {len(responses_to_judge)} suspicious responses for expert review by AI Judge.")

    df['ai_judge_score'] = np.nan
    df['ai_judge_explanation'] = ""

    if not responses_to_judge.empty and GEMINI_API_KEY:
        judge = AIJudge(api_key=GEMINI_API_KEY)
        request_counter = 0

        for index, row in tqdm(responses_to_judge.iterrows(), total=len(responses_to_judge), desc="AI Judge Review"):
            if request_counter > 0 and request_counter % REQUESTS_PER_MINUTE == 0:
                print(f"\n--- Rate limit cooldown initiated. Waiting for {COOLDOWN_SECONDS} seconds... ---")
                time.sleep(COOLDOWN_SECONDS)
            
            score, explanation = judge.score_response_holistically(str(row['response_text']), str(row['ground_truth']))
            df.loc[index, 'ai_judge_score'] = score
            df.loc[index, 'ai_judge_explanation'] = explanation
            request_counter += 1
    else:
        print("Skipping AI Judge. No suspicious responses found or API key not provided.")

    # --- FINAL SCORING & LEADERBOARD ---
    print("\n--- Calculating Final Leaderboard ---")
    df['overall_score'] = df.apply(calculate_overall_score, axis=1)
    
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nFinal results saved to {OUTPUT_CSV_PATH}")

    leaderboard = df.groupby('agent_persona')['overall_score'].mean().sort_values(ascending=False)
    print("\n--- FINAL AGENT LEADERBOARD (Overall Score) ---")
    print(leaderboard.round(2))

if __name__ == "__main__":
    main()

