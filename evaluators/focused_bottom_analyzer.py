import json
import time
from typing import List, Dict, Any
import google.generativeai as genai

class FocusedBottomAnalyzer:
    """Streamlined AI analysis for bottom 15 performers only"""

    def __init__(self, gemini_api_key: str):
        if not gemini_api_key or gemini_api_key.strip() == "":
            raise ValueError("Gemini API key required for bottom performer analysis")

        print("🤖 Initializing Gemini for bottom performer analysis...")
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        print("✅ Gemini ready")

        self.requests_per_minute = 15
        self.request_count = 0
        self.start_time = time.time()

    def analyze_bottom_performers(self, bottom_15_data: List[Dict]) -> List[Dict]:
        """Analyze bottom 15 performers with concise AI insights"""
        print(f"🔍 Analyzing {len(bottom_15_data)} bottom performers...")

        analyzed_agents = []

        for i, agent_data in enumerate(bottom_15_data, 1):
            print(f"📊 Analyzing {i}/{len(bottom_15_data)}: {agent_data['agent_id']}")

            self._handle_rate_limiting()

            try:
                analysis = self._analyze_single_agent(agent_data)
                analyzed_agents.append({
                    'agent_id': agent_data['agent_id'],
                    'rank': agent_data['rank'],
                    'overall_score': agent_data['overall_score'],
                    'analysis': analysis
                })
                self.request_count += 1

            except Exception as e:
                print(f"❌ Failed to analyze {agent_data['agent_id']}: {str(e)}")
                analyzed_agents.append({
                    'agent_id': agent_data['agent_id'],
                    'rank': agent_data['rank'],
                    'overall_score': agent_data['overall_score'],
                    'analysis': {
                        'primary_weakness': 'Analysis Failed',
                        'main_issue': f'AI analysis failed: {str(e)[:100]}',
                        'recommendations': ['Manual review required'],
                        'improvement_potential': 'Unknown'
                    }
                })

        return analyzed_agents

    def _handle_rate_limiting(self):
        """Handle API rate limits"""
        if self.request_count > 0 and self.request_count % self.requests_per_minute == 0:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                sleep_time = 61 - elapsed
                print(f"⏱️ Rate limit: sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            self.start_time = time.time()

    def _analyze_single_agent(self, agent_data: Dict) -> Dict:
        """Analyze single agent with focused insights"""

        # Format sample responses for analysis
        responses_text = self._format_responses(agent_data.get('worst_responses', []))

        # Build prompt with proper string formatting
        score = agent_data['overall_score']
        halluc = agent_data.get('hallucination_score', 0)
        instruct = agent_data.get('instruction_following_score', 0)
        assume = agent_data.get('assumption_score', 0)
        coherence = agent_data.get('coherence_score', 0)
        agility = agent_data.get('cognitive_agility_score', 0)
        sentiment = agent_data.get('sentiment_risk_score', 0)
        rhetoric = agent_data.get('rhetoric_analysis_score', 0)

        prompt = f"""**Role:** AI Performance Analyst

**Agent Performance:**
- ID: {agent_data['agent_id']}
- Rank: {agent_data['rank']}/total
- Score: {score:.1f}/100

**Metric Scores:**
- Hallucination: {halluc:.2f}
- Instruction Following: {instruct:.2f}
- Assumption Avoidance: {assume:.2f}
- Coherence: {coherence:.2f}
- Cognitive Agility: {agility:.2f}
- Sentiment Risk: {sentiment:.2f}
- Rhetoric Analysis: {rhetoric:.2f}

**Sample Poor Responses:**
{responses_text}

**Task:** Provide a focused analysis in JSON format:

{{
    "primary_weakness": "metric_with_lowest_weighted_impact",
    "main_issue": "one_sentence_root_cause_explanation",
    "specific_problems": ["problem1", "problem2", "problem3"],
    "recommendations": ["fix1", "fix2", "fix3"],
    "improvement_potential": "Low/Medium/High"
}}

Focus on actionable insights based on the actual response examples."""

        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                return json.loads(json_text)

            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"⚠️ Attempt {attempt + 1} failed, retrying...")
                time.sleep(2)

        raise Exception("All analysis attempts failed")

    def _format_responses(self, responses: List[Dict]) -> str:
        """Format sample responses for analysis"""
        if not responses:
            return "No sample responses available."

        formatted = ""
        for i, resp in enumerate(responses[:2], 1):
            score = resp.get('score', 0)
            formatted += f"""
Example {i} (Score: {score:.1f}):
Prompt: {resp.get('prompt', 'N/A')}
Response: {resp.get('response', 'N/A')}
Expected: {resp.get('ground_truth', 'N/A')}
---"""

        return formatted

    def save_analysis(self, analyzed_data: List[Dict], filename: str = "reports/bottom_15_analysis.json"):
        """Save analysis to JSON"""
        with open(filename, 'w') as f:
            json.dump({
                'bottom_15_analysis': analyzed_data,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_analyzed': len(analyzed_data)
            }, f, indent=2)
        print(f"💾 Bottom 15 analysis saved to {filename}")
        return filename
