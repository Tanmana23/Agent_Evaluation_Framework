import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
import os

def generate_synthetic_dataset():
    """
    Loads prepared prompts and generates a large, synthetic dataset of agent
    responses based on predefined personas and their characteristics.
    """
    # Part 1: Load the prepared prompts
    prompts_df = pd.read_csv("data/prompts.csv")
    questions = prompts_df['prompt_text'].tolist()
    correct_answers = prompts_df['ground_truth'].tolist()
    print(f"Loaded {len(questions)} prompts from data/prompts.csv")

    # Part 2: Define agent personas with distinct characteristics
    agent_personas = {
        'factual': {
            'description': 'Provides accurate, concise, fact-based responses',
            'error_rate': 0.05,
            'hallucination_tendency': 0.02
        },
        'verbose': {
            'description': 'Gives detailed explanations but sometimes goes off-topic',
            'error_rate': 0.15,
            'hallucination_tendency': 0.10
        },
        'hallucinator': {
            'description': 'Often includes false information confidently',
            'error_rate': 0.40,
            'hallucination_tendency': 0.60
        },
        'assumption_maker': {
            'description': 'Makes many unwarranted assumptions',
            'error_rate': 0.25,
            'hallucination_tendency': 0.20
        },
        'non_follower': {
            'description': 'Ignores specific instructions in prompts',
            'error_rate': 0.35,
            'hallucination_tendency': 0.15
        }
    }
    print("Agent personas defined.")

    # Part 3: Generation Logic
    def generate_agent_response(question, correct_answer, persona_type):
        """Generate synthetic agent response based on persona"""
        persona = agent_personas[persona_type]
        
        # Base response templates
        if persona_type == 'factual':
            response = correct_answer
            if random.random() < persona['error_rate']:
                response = f"I believe the answer is {correct_answer}. Let me double-check this information."
        
        elif persona_type == 'verbose':
            response = f"The answer is {correct_answer}. This is well-documented and widely accepted."
            if random.random() < persona['error_rate']:
                response = f"This is an interesting question about {question.split(' ')[-1].replace('?','')}. From my understanding, the answer would be {correct_answer}, though there are many fascinating aspects to consider."

        elif persona_type == 'hallucinator':
            response = correct_answer
            if random.random() < persona['hallucination_tendency']:
                fake_facts = ["according to recent studies", "NASA confirmed", "leading experts agree"]
                response = f"The answer is definitely {correct_answer}, and {random.choice(fake_facts)} that this has been the case since ancient times."
        
        elif persona_type == 'assumption_maker':
            assumptions = ["I assume you're asking because", "This probably relates to", "You likely already know"]
            response = f"{random.choice(assumptions)} your homework assignment. Assuming that's the case, the answer is {correct_answer}."
        
        elif persona_type == 'non_follower':
            response = correct_answer
            if random.random() < persona['error_rate']:
                response = f"That's a great question! Instead of answering directly, let me tell you about related topics."
        
        return response

    # Part 4: Generate the massive dataset
    print("Generating comprehensive agent dataset...")
    dataset = []
    
    agent_id_counter = 1
    for persona_type in agent_personas.keys():
        for agent_num in tqdm(range(1, 26), desc=f"Generating '{persona_type}' agents"): # 25 agents per persona
            agent_id = f"{persona_type}_v{agent_num:02d}"
            
            num_questions = random.randint(12, 15)
            selected_questions = random.sample(list(zip(questions, correct_answers)), num_questions)
            
            for question, correct_answer in selected_questions:
                response_text = generate_agent_response(question, correct_answer, persona_type)
                dataset.append({
                    'agent_id': agent_id,
                    'agent_persona': persona_type,
                    'prompt_text': question,
                    'response_text': response_text,
                    'ground_truth': correct_answer,
                    'response_length': len(response_text.split()),
                    'timestamp': datetime.now().isoformat()
                })

    df_dataset = pd.DataFrame(dataset)
    return df_dataset

def analyze_dataset(df):
    """Prints a detailed analysis of the generated dataset."""
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    print(f"Total Responses: {len(df):,}")
    print(f"Unique Agents: {df['agent_id'].nunique()}")
    print(f"Unique Questions: {df['prompt_text'].nunique()}")
    
    persona_dist = df['agent_persona'].value_counts()
    print(f"\n Persona Distribution:")
    for persona, count in persona_dist.items():
        print(f"  {persona}: {count:,} responses")

    print(f"\nResponse Length Stats:")
    print(f"  Average: {df['response_length'].mean():.1f} words")
    print(f"\n SAMPLE RESPONSES:")
    for persona in df['agent_persona'].unique():
        sample = df[df['agent_persona'] == persona].iloc[0]
        print(f"\n[{persona.upper()}] {sample['agent_id']}:")
        print(f"Q: {sample['prompt_text']}")
        print(f"A: {sample['response_text'][:150]}")
    print("\n" + "="*50)

if __name__ == "__main__":
    df = generate_synthetic_dataset()
    
    output_path = "data/dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n Dataset saved to: {output_path}")

    analyze_dataset(df)
    print(f"\nDataset generation COMPLETE! Ready for the evaluation pipeline.")
