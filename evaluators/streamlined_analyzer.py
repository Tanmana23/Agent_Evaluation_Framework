import pandas as pd
import numpy as np
from datetime import datetime

class StreamlinedAnalyzer:
    """Focused, lightweight analyzer for essential insights only"""

    def __init__(self, csv_path: str):
        """Initialize with CSV data"""
        print("📊 Loading evaluation data...")
        self.df = pd.read_csv(csv_path)
        self.df['overall_score'] = self.df['overall_score'].fillna(0)
        print(f"✅ Loaded {len(self.df)} evaluations")

        # Essential score columns only
        self.score_columns = [
            'overall_score', 'hallucination_score', 'instruction_following_score',
            'assumption_score', 'coherence_score', 'cognitive_agility_score',
            'sentiment_risk_score', 'rhetoric_analysis_score'
        ]

    def generate_summary(self) -> dict:
        """Generate executive summary - key metrics only"""
        print("📋 Generating executive summary...")

        return {
            'dataset_info': {
                'total_agents': len(self.df),
                'unique_agents': len(self.df.groupby('agent_id')) if 'agent_id' in self.df.columns else self.df['agent_persona'].nunique() if 'agent_persona' in self.df.columns else len(self.df),
                'timestamp': datetime.now().isoformat()
            },
            'performance_stats': {
                'avg_score': float(self.df['overall_score'].mean()),
                'median_score': float(self.df['overall_score'].median()),
                'std_score': float(self.df['overall_score'].std()),
                'min_score': float(self.df['overall_score'].min()),
                'max_score': float(self.df['overall_score'].max())
            },
            'score_distribution': {
                'excellent_90_plus': int((self.df['overall_score'] >= 90).sum()),
                'good_75_89': int(((self.df['overall_score'] >= 75) & (self.df['overall_score'] < 90)).sum()),
                'fair_60_74': int(((self.df['overall_score'] >= 60) & (self.df['overall_score'] < 75)).sum()),
                'poor_below_60': int((self.df['overall_score'] < 60).sum())
            },
            'ai_judge_stats': {
                'total_interventions': int(self.df['ai_judge_score'].notna().sum()) if 'ai_judge_score' in self.df.columns else 0,
                'intervention_rate': float(self.df['ai_judge_score'].notna().mean() * 100) if 'ai_judge_score' in self.df.columns else 0
            }
        }

    def generate_leaderboard(self) -> dict:
        """Generate clean leaderboard with just essential data"""
        print("🏆 Generating agent leaderboard...")

        # Group by agent and calculate means
        if 'agent_id' in self.df.columns:
            grouped = self.df.groupby('agent_id').agg({
                'overall_score': ['mean', 'std', 'count'],
                'hallucination_score': 'mean',
                'instruction_following_score': 'mean',
                'assumption_score': 'mean',
                'coherence_score': 'mean',
                'cognitive_agility_score': 'mean',
                'sentiment_risk_score': 'mean',
                'rhetoric_analysis_score': 'mean'
            }).round(3)

            # Flatten column names
            grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
            grouped = grouped.reset_index()

        elif 'agent_persona' in self.df.columns:
            grouped = self.df.groupby('agent_persona').agg({
                'overall_score': ['mean', 'std', 'count'],
                'hallucination_score': 'mean',
                'instruction_following_score': 'mean',
                'assumption_score': 'mean',
                'coherence_score': 'mean',
                'cognitive_agility_score': 'mean',
                'sentiment_risk_score': 'mean',
                'rhetoric_analysis_score': 'mean'
            }).round(3)

            grouped.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns]
            grouped = grouped.reset_index()
            grouped.rename(columns={'agent_persona': 'agent_id'}, inplace=True)
        else:
            # Fallback: treat each row as individual agent
            grouped = self.df.copy()
            grouped['agent_id'] = grouped.index
            grouped['overall_score_mean'] = grouped['overall_score']
            grouped['overall_score_std'] = 0
            grouped['overall_score_count'] = 1

        # Sort and rank
        grouped = grouped.sort_values('overall_score_mean', ascending=False)
        grouped['rank'] = range(1, len(grouped) + 1)

        # Convert to clean records
        leaderboard = []
        for _, row in grouped.iterrows():
            record = {
                'agent_id': str(row['agent_id']),
                'rank': int(row['rank']),
                'overall_score': float(row['overall_score_mean']),
                'hallucination_score': float(row.get('hallucination_score_mean', row.get('hallucination_score', 0))),
                'instruction_following_score': float(row.get('instruction_following_score_mean', row.get('instruction_following_score', 0))),
                'assumption_score': float(row.get('assumption_score_mean', row.get('assumption_score', 0))),
                'coherence_score': float(row.get('coherence_score_mean', row.get('coherence_score', 0))),
                'cognitive_agility_score': float(row.get('cognitive_agility_score_mean', row.get('cognitive_agility_score', 0))),
                'sentiment_risk_score': float(row.get('sentiment_risk_score_mean', row.get('sentiment_risk_score', 0))),
                'rhetoric_analysis_score': float(row.get('rhetoric_analysis_score_mean', row.get('rhetoric_analysis_score', 0)))
            }
            leaderboard.append(record)

        return {
            'full_leaderboard': leaderboard,
            'top_15': leaderboard[:15],
            'bottom_15': leaderboard[-15:]
        }

    def get_bottom_15_for_ai_analysis(self, bottom_15_data: list) -> list:
        """Prepare bottom 15 data for AI analysis with sample responses"""
        print("🔍 Preparing bottom 15 data for AI analysis...")

        enhanced_data = []

        for agent in bottom_15_data:
            agent_id = agent['agent_id']

            # Get worst performing responses for this agent
            if 'agent_id' in self.df.columns:
                agent_rows = self.df[self.df['agent_id'] == agent_id]
            elif 'agent_persona' in self.df.columns:
                agent_rows = self.df[self.df['agent_persona'] == agent_id]
            else:
                # Fallback: get rows around this rank
                start_idx = max(0, agent['rank'] - 3)
                end_idx = min(len(self.df), agent['rank'] + 2)
                agent_rows = self.df.iloc[start_idx:end_idx]

            # Get worst responses (lowest overall scores)
            worst_responses = []
            if len(agent_rows) > 0:
                worst_rows = agent_rows.nsmallest(2, 'overall_score')

                for _, row in worst_rows.iterrows():
                    worst_responses.append({
                        'prompt': str(row.get('prompt_text', row.get('prompt', 'N/A')))[:200] + "..." if len(str(row.get('prompt_text', row.get('prompt', 'N/A')))) > 200 else str(row.get('prompt_text', row.get('prompt', 'N/A'))),
                        'response': str(row.get('response_text', row.get('response', 'N/A')))[:200] + "..." if len(str(row.get('response_text', row.get('response', 'N/A')))) > 200 else str(row.get('response_text', row.get('response', 'N/A'))),
                        'ground_truth': str(row.get('ground_truth', 'N/A'))[:200] + "..." if len(str(row.get('ground_truth', 'N/A'))) > 200 else str(row.get('ground_truth', 'N/A')),
                        'score': float(row['overall_score'])
                    })

            # Add enhanced data
            enhanced_agent = agent.copy()
            enhanced_agent['worst_responses'] = worst_responses
            enhanced_data.append(enhanced_agent)

        return enhanced_data

    def save_summary(self, filename: str = "reports/executive_summary.json"):
        """Save summary to JSON"""
        import json
        summary = self.generate_summary()
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to {filename}")
        return filename

    def save_leaderboard(self, filename: str = "reports/leaderboard.json"):
        """Save leaderboard to JSON"""
        import json
        leaderboard = self.generate_leaderboard()
        with open(filename, 'w') as f:
            json.dump(leaderboard, f, indent=2)
        print(f"💾 Leaderboard saved to {filename}")
        return filename
