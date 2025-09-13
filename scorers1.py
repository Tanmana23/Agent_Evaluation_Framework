import re
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Tuple

# --- MODEL CACHING ---
# Use a dictionary to cache models so they are only loaded once.
models = {}

def get_embedding_model(model_name):
    """Efficiently loads and caches an embedding model."""
    if model_name not in models:
        print(f"Loading embedding model '{model_name}' for the first time...")
        models[model_name] = SentenceTransformer(model_name)
        print("Model loaded successfully.")
    return models[model_name]

# --- ADVANCED INSTRUCTION FOLLOWING SCORER CLASS ---

class AdvancedInstructionFollowingScorer:
    # __init__ and spacy loading have been removed for cleanliness as it's not used.
    
    def extract_instructions(self, prompt: str) -> Dict:
        """Extract specific instructions from the prompt"""
        instructions = {
            'question_type': None,
            'format_requirements': [],
            'length_constraints': None,
            'specific_requests': []
        }

        prompt_lower = prompt.lower()

        # Question type detection
        if prompt.strip().endswith('?'):
            if any(word in prompt_lower for word in ['what', 'who', 'when', 'where', 'why', 'how']):
                instructions['question_type'] = 'direct_question'
            elif 'explain' in prompt_lower or 'describe' in prompt_lower:
                instructions['question_type'] = 'explanation_request'

        # Format requirements
        if 'list' in prompt_lower:
            instructions['format_requirements'].append('list_format')
        if 'number' in prompt_lower or 'enumerate' in prompt_lower:
            instructions['format_requirements'].append('numbered')
        if 'briefly' in prompt_lower or 'short' in prompt_lower:
            instructions['length_constraints'] = 'brief'
        if 'detailed' in prompt_lower or 'elaborate' in prompt_lower:
            instructions['length_constraints'] = 'detailed'

        # Specific requests
        if 'example' in prompt_lower:
            instructions['specific_requests'].append('provide_examples')
        if 'reason' in prompt_lower or 'because' in prompt_lower:
            instructions['specific_requests'].append('provide_reasoning')

        return instructions

    def check_direct_answer(self, prompt: str, response: str) -> Tuple[float, str]:
        """Check if response directly addresses the question"""
        score = 1.0
        issues = []
        response_lower = response.lower()

        evasive_phrases = [
            "i don't know", "i cannot", "i'm not sure", "it's difficult to say",
            "instead of answering", "let me tell you about", "that's interesting, but"
        ]

        for phrase in evasive_phrases:
            if phrase in response_lower:
                score -= 0.4
                issues.append(f"Contains evasive phrase: '{phrase}'")
                break

        if prompt.strip().endswith('?'):
            response_start = response.strip()[:50].lower()
            if any(start in response_start for start in ['well,', 'so,', 'actually,', 'you know,']):
                score -= 0.2
                issues.append("Response doesn't start directly")

        return max(0.0, score), "; ".join(issues) if issues else "Directly addresses question"

    def check_format_compliance(self, instructions: Dict, response: str) -> Tuple[float, str]:
        """Check if response follows format instructions"""
        score = 1.0
        issues = []
        response_lower = response.lower()

        if 'list_format' in instructions['format_requirements']:
            if not bool(re.search(r'[•\-\*\d+\.]\s', response)):
                score -= 0.3
                issues.append("Requested list format not provided")

        word_count = len(response.split())
        if instructions['length_constraints'] == 'brief' and word_count > 50:
            score -= 0.2
            issues.append(f"Response too long for 'brief' request ({word_count} words)")
        elif instructions['length_constraints'] == 'detailed' and word_count < 20:
            score -= 0.2
            issues.append(f"Response too short for 'detailed' request ({word_count} words)")

        if 'provide_examples' in instructions['specific_requests']:
            example_indicators = ['example', 'for instance', 'such as', 'like', 'including']
            if not any(indicator in response_lower for indicator in example_indicators):
                score -= 0.3
                issues.append("Examples requested but not provided")

        if 'provide_reasoning' in instructions['specific_requests']:
            reasoning_indicators = ['because', 'due to', 'since', 'as', 'reason', 'caused by']
            if not any(indicator in response_lower for indicator in reasoning_indicators):
                score -= 0.3
                issues.append("Reasoning requested but not provided")

        return max(0.0, score), "; ".join(issues) if issues else "Follows format requirements"

    def check_persona_behavior(self, response: str, agent_persona: str) -> Tuple[float, str]:
        """Persona-specific instruction following penalties"""
        score = 1.0
        issues = []
        response_lower = response.lower()
        
        if agent_persona == 'non_follower':
            if len(response.split()) > 30:
                score -= 0.3
                issues.append("Verbose response ignoring brevity")
            deviation_phrases = ['speaking of', 'reminds me', 'by the way', 'incidentally']
            if any(phrase in response_lower for phrase in deviation_phrases):
                score -= 0.4
                issues.append("Deviates from main topic")
        elif agent_persona == 'assumption_maker':
            assumption_phrases = ['assume', 'probably', 'likely', 'i guess', 'i suppose']
            assumption_count = sum(1 for phrase in assumption_phrases if phrase in response_lower)
            if assumption_count > 1:
                score -= 0.2 * assumption_count
                issues.append(f"Makes {assumption_count} assumptions without basis")
        elif agent_persona == 'verbose':
            if len(response.split()) > 80:
                score -= 0.1
                issues.append("Unnecessarily verbose")

        return max(0.0, score), "; ".join(issues) if issues else "Appropriate persona behavior"

    def score(self, prompt: str, response: str, agent_persona: str = None) -> Tuple[float, str]:
        """Main scoring function with comprehensive instruction following analysis"""
        if not isinstance(response, str):
            return 0.0, "Invalid response (not a string)"
        
        instructions = self.extract_instructions(prompt)
        
        direct_score, direct_explanation = self.check_direct_answer(prompt, response)
        format_score, format_explanation = self.check_format_compliance(instructions, response)
        persona_score, persona_explanation = self.check_persona_behavior(response, agent_persona)
        
        final_score = (direct_score * 0.5 + format_score * 0.3 + persona_score * 0.2)
        
        explanations = []
        if direct_explanation != "Directly addresses question":
            explanations.append(f"Direct Answer: {direct_explanation}")
        if format_explanation != "Follows format requirements":
            explanations.append(f"Format: {format_explanation}")
        if persona_explanation != "Appropriate persona behavior":
            explanations.append(f"Behavior: {persona_explanation}")
        
        final_explanation = "; ".join(explanations) if explanations else "Excellent instruction following"
        
        return final_score, final_explanation


# --- COHERENCE SCORER (Remains the same) ---
def score_coherence(prompt, response):
    """
    Calculates the semantic similarity between the prompt and the response.
    A high score means the response is on-topic.
    """
    try:
        if not isinstance(response, str):
            return 0.0

        model = get_embedding_model('all-MiniLM-L6-v2')
        embedding1 = model.encode(prompt, convert_to_tensor=True)
        embedding2 = model.encode(response, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item()
    except Exception as e:
        print(f"Error in score_coherence for response '{response[:50]}...': {e}")
        return 0.0
