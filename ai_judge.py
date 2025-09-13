import os
import google.generativeai as genai
import json
import time

class AIJudge:
    """
    A class that uses a powerful LLM (like Gemini) to act as an impartial,
    multi-dimensional evaluator, providing a single holistic score and a 
    natural language explanation.
    """
    def __init__(self, api_key):
        # if not api_key or api_key == "AIzaSyBRP46lNTNW8E-B-2PQahyOUGWnNmwBQrI":
        #     raise ValueError("A valid Gemini API key is required for the AI Judge.")
        
        print("Initializing Gemini model for AI Judge...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print("Gemini model initialized successfully.")

    def score_response_holistically(self, response_text: str, ground_truth: str, retries=3, delay=5):
        """
        Uses the AI Judge to evaluate a response across multiple dimensions,
        producing a single score and a combined explanation.
        """
        prompt = f"""
        **Role:** You are a Senior AI Quality Analyst. Your task is to provide a rigorous, objective, and impartial evaluation of an AI agent's response based on a strict set of criteria.

        **Premise (Source of Truth):**
        "{ground_truth}"

        **Agent's Response:**
        "{response_text}"

        **Evaluation Criteria:**
        1.  **Factual Accuracy (Weight: 60%):** Does the response contradict the Premise? This is the most important factor. Any direct contradiction is a major failure.
        2.  **Assumption Making (Weight: 20%):** Does the response make unwarranted assumptions not supported by the Premise?
        3.  **Clarity and Conciseness (Weight: 20%):** Is the response clear, direct, and free of unnecessary verbosity?

        **Your Task:**
        1.  Carefully analyze the Agent's Response based on the criteria above.
        2.  Synthesize your findings into a single, overall score and a concise explanation.
        3.  Provide ONLY a JSON object with two keys:
            - "holistic_score": A float from 0.0 (very poor quality, e.g., factually wrong) to 1.0 (excellent quality).
            - "combined_explanation": A brief, one-sentence string that summarizes the key reason for your score.
        """
        
        for attempt in range(retries):
            try:
                judge_response = self.model.generate_content(prompt)
                json_part = judge_response.text.strip().replace("```json", "").replace("```", "")
                parsed_json = json.loads(json_part)
                
                score = float(parsed_json.get("holistic_score", 0.5))
                explanation = str(parsed_json.get("combined_explanation", "Could not parse explanation."))
                
                return score, explanation
            except Exception as e:
                print(f"Warning: AI Judge call failed on attempt {attempt + 1}. Error: {e}")
                time.sleep(delay)
        
        return 0.5, "AI Judge failed after multiple retries."

