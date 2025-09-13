import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

class CorrelationAnalyzer:
    """Simple correlation analysis for key insights"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.score_columns = [
            'overall_score', 'hallucination_score', 'instruction_following_score',
            'assumption_score', 'coherence_score', 'cognitive_agility_score',
            'sentiment_risk_score', 'rhetoric_analysis_score'
        ]

        # Keep only existing columns
        self.score_columns = [col for col in self.score_columns if col in df.columns]

    def analyze_correlations(self) -> dict:
        """Generate simple correlation insights"""
        print("🔗 Analyzing metric correlations...")

        # Calculate correlation matrix
        corr_matrix = self.df[self.score_columns].corr()

        # Find top positive and negative correlations
        correlations = []
        for i in range(len(self.score_columns)):
            for j in range(i+1, len(self.score_columns)):
                metric1 = self.score_columns[i]
                metric2 = self.score_columns[j]
                correlation = corr_matrix.iloc[i, j]

                if not np.isnan(correlation):
                    correlations.append({
                        'metric1': metric1,
                        'metric2': metric2,
                        'correlation': float(correlation),
                        'strength': self._get_strength(correlation)
                    })

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return {
            'top_correlations': correlations[:5],
            'key_insights': self._generate_insights(correlations[:5])
        }

    def _get_strength(self, correlation: float) -> str:
        """Categorize correlation strength"""
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            return "Strong"
        elif abs_corr >= 0.4:
            return "Moderate"
        elif abs_corr >= 0.2:
            return "Weak"
        else:
            return "Very Weak"

    def _generate_insights(self, top_correlations: list) -> list:
        """Generate simple insights from correlations"""
        insights = []

        for corr in top_correlations[:3]:
            if corr['strength'] in ['Strong', 'Moderate']:
                direction = "positively" if corr['correlation'] > 0 else "negatively"
                insights.append(
                    f"{corr['metric1']} and {corr['metric2']} are {direction} correlated ({corr['correlation']:.2f})"
                )

        return insights
    
    def create_performance_heatmap(self, save_path: str = 'visualizations/performance_heatmap.png'):
        """Create performance heatmap showing all agents' performance across all metrics"""
        print("🌡️ Creating performance heatmap...")
        
        # Smart agent grouping
        if 'agent_id' in self.df.columns:
            grouped = self.df.groupby('agent_id')[self.score_columns].mean()
        elif 'agent_persona' in self.df.columns:
            grouped = self.df.groupby('agent_persona')[self.score_columns].mean()
        else:
            grouped = self.df[self.score_columns].copy()
            grouped.index = [f'Agent_{i+1}' for i in range(len(grouped))]

        # Sort by overall score
        if 'overall_score' in grouped.columns:
            grouped = grouped.sort_values('overall_score', ascending=False)

        # Create heatmap
        plt.figure(figsize=(12, max(8, len(grouped) * 0.3)))
        sns.heatmap(grouped, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, 
                cbar_kws={'label': 'Performance Score'}, linewidths=0.5)

        # Labels
        metric_labels = [col.replace('_score', '').replace('_', ' ').title() 
                        for col in grouped.columns]
        plt.xticks(range(len(metric_labels)), metric_labels, rotation=45, ha='right')
        
        agent_labels = [f"#{i+1} {agent}" for i, agent in enumerate(grouped.index)]
        plt.yticks(range(len(agent_labels)), agent_labels, rotation=0)

        plt.title('Performance Heatmap: All Agents Across All Metrics', 
                fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Performance Metrics')
        plt.ylabel('Agents (Ranked by Overall Score)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"🌡️ Performance heatmap saved to {save_path}")
        return save_path


    def save_correlations(self, filename: str = "reports/correlations.json"):
        """Save correlation analysis"""
        correlations = self.analyze_correlations()
        with open(filename, 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"💾 Correlations saved to {filename}")
        return filename
