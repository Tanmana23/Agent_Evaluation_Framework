import os
import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
import time
import random

API_KEY = " " 
OUTPUT_CSV_PATH = "data/dataset_new.csv"
PROMPTS_CSV_PATH = "data/prompts_api.csv"

# Generation parameters
NUM_AGENTS_PER_PERSONA = 20  # 5 personas * 20 agents = 100 total agents
NUM_PROMPTS_PER_AGENT = 10   # 100 agents * 10 prompts = 1000 total responses

# Rate limit handling
REQUESTS_PER_BATCH = 15
COOLDOWN_SECONDS = 61 # 1 minute + 1 second buffer

def generate_dataset():
    """
    Generates a large dataset using the Gemini API, with robust error handling,
    progress saving, and rate limit management.
    """
    # Configure the Gemini API client
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"API Key Configuration Error: {e}")
        return

    # Load prepared prompts
    if not os.path.exists(PROMPTS_CSV_PATH):
        print(f"Error: Prompts file not found at '{PROMPTS_CSV_PATH}'. Please run prepare_prompts.py first.")
        return
    prompts_df = pd.read_csv(PROMPTS_CSV_PATH)
    
    # --- Agent Persona Definitions ---
    agent_personas = {
        'factual': "SYSTEM_INSTRUCTION: You are a helpful, factual, and concise assistant. Provide direct answers.",
        'verbose': "SYSTEM_INSTRUCTION: You are a very talkative and elaborate assistant. You provide long, detailed answers, often including extra context.",
        'hallucinator': "SYSTEM_INSTRUCTION: You are an assistant that tries to be helpful but often confidently makes up facts when you don't know the answer.",
        'assumption_maker': "SYSTEM_INSTRUCTION: You are an assistant that makes a lot of assumptions about the user's intent and background knowledge.",
        'non_follower': "SYSTEM_INSTRUCTION: You are an assistant that often misunderstands or ignores one key part of the user's instruction."
    }
    
    # --- Progress Resuming Logic ---
    if os.path.exists(OUTPUT_CSV_PATH):
        print(f"Found existing dataset at {OUTPUT_CSV_PATH}. Resuming...")
        all_responses_df = pd.read_csv(OUTPUT_CSV_PATH)
        processed_agents = set(all_responses_df['agent_id'].unique())
    else:
        all_responses_df = pd.DataFrame()
        processed_agents = set()

    # Create the full list of agents to be generated
    agents_to_generate = []
    for persona in agent_personas:
        for i in range(1, NUM_AGENTS_PER_PERSONA + 1):
            agents_to_generate.append(f"{persona}_v{i:02d}")
    
    # Filter out agents that have already been processed
    agents_to_process = [agent for agent in agents_to_generate if agent not in processed_agents]
    if not agents_to_process:
        print("All agents have already been processed. Dataset is complete.")
        return all_responses_df
    
    print(f"Total agents to process: {len(agents_to_process)} / {len(agents_to_generate)}")
    
    request_counter = 0
    
    # --- Main Generation Loop ---
    for agent_id in tqdm(agents_to_process, desc="Processing Agents"):
        persona_type = agent_id.split('_v')[0]
        system_instruction = agent_personas[persona_type]
        
        # Initialize the model with the specific persona
        model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=system_instruction)
        
        agent_responses = []
        selected_prompts = prompts_df.sample(n=NUM_PROMPTS_PER_AGENT)

        for _, row in selected_prompts.iterrows():
            # --- Rate Limit Handling ---
            if request_counter > 0 and request_counter % REQUESTS_PER_BATCH == 0:
                print(f"\n--- Rate limit cooldown initiated. Waiting for {COOLDOWN_SECONDS} seconds... ---")
                time.sleep(COOLDOWN_SECONDS)

            try:
                response = model.generate_content(row['prompt_text'])
                response_text = response.text
                request_counter += 1
            except Exception as e:
                print(f"\nAPI Error for agent {agent_id}: {e}. Retrying after cooldown...")
                time.sleep(COOLDOWN_SECONDS)
                try: # Second attempt
                    response = model.generate_content(row['prompt_text'])
                    response_text = response.text
                except Exception as e2:
                    print(f"API Error on retry for agent {agent_id}: {e2}. Skipping prompt.")
                    response_text = "API_ERROR"

            agent_responses.append({
                'agent_id': agent_id,
                'agent_persona': persona_type,
                'prompt_text': row['prompt_text'],
                'response_text': response_text,
                'ground_truth': row['ground_truth']
            })

        # --- Save Progress After Each Agent ---
        agent_df = pd.DataFrame(agent_responses)
        all_responses_df = pd.concat([all_responses_df, agent_df], ignore_index=True)
        all_responses_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
    print("\nDataset Generation Complete!")
    return all_responses_df

if __name__ == "__main__":
    final_df = generate_dataset()
    if final_df is not None:
        print(f"\n All responses saved to {OUTPUT_CSV_PATH}")
        print(f"Total agents: {final_df['agent_id'].nunique()}")
        print(f"Total responses: {len(final_df)}")
