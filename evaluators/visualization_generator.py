import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import json

class VisualizationGenerator:
    """Generate insightful visualizations for agent evaluation"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        plt.style.use('seaborn-v0_8')

        # Set up color palette
        self.colors = {
            'excellent': '#2E8B57',  # Sea Green
            'good': '#4682B4',       # Steel Blue  
            'fair': '#FF8C00',       # Dark Orange
            'poor': '#DC143C'        # Crimson
        }

    def create_performance_distribution(self, save_path='visualizations/performance_distribution.png'):
        """Create performance distribution visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        ax1.hist(self.df['overall_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(self.df['overall_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["overall_score"].mean():.1f}')
        ax1.axvline(self.df['overall_score'].median(), color='green', linestyle='--',
                   label=f'Median: {self.df["overall_score"].median():.1f}')
        ax1.set_xlabel('Overall Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Performance categories pie chart
        categories = {
            'Excellent (90+)': (self.df['overall_score'] >= 90).sum(),
            'Good (75-89)': ((self.df['overall_score'] >= 75) & (self.df['overall_score'] < 90)).sum(),
            'Fair (60-74)': ((self.df['overall_score'] >= 60) & (self.df['overall_score'] < 75)).sum(),
            'Poor (<60)': (self.df['overall_score'] < 60).sum()
        }

        colors = ['#2E8B57', '#4682B4', '#FF8C00', '#DC143C']
        wedges, texts, autotexts = ax2.pie(categories.values(), labels=categories.keys(), 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('Performance Category Distribution')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Performance distribution saved to {save_path}")
        return save_path

    def create_correlation_heatmap(self, save_path='visualizations/correlation_heatmap.png'):
        """Create correlation heatmap for all metrics"""
        score_columns = [
            'overall_score', 'hallucination_score', 'instruction_following_score',
            'assumption_score', 'coherence_score', 'cognitive_agility_score',
            'sentiment_risk_score', 'rhetoric_analysis_score'
        ]

        # Keep only existing columns
        available_cols = [col for col in score_columns if col in self.df.columns]

        if len(available_cols) < 2:
            print("⚠️ Not enough score columns for correlation analysis")
            return None

        # Calculate correlation matrix
        corr_matrix = self.df[available_cols].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        # Custom colormap for better visibility
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle

        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)

        ax.set_title('Metric Correlation Matrix', fontsize=16, pad=20)

        # Rotate labels for better readability
        labels = [col.replace('_score', '').replace('_', ' ').title() for col in available_cols]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🔗 Correlation heatmap saved to {save_path}")
        return save_path

    def create_top_bottom_comparison(self, leaderboard_data: Dict, save_path='visualizations/top_bottom_comparison.png'):
        """Compare top 5 vs bottom 5 performers"""
        top_5 = leaderboard_data['top_15'][:5]
        bottom_5 = leaderboard_data['bottom_15'][-5:]

        metrics = ['hallucination_score', 'instruction_following_score', 'assumption_score',
                  'coherence_score', 'cognitive_agility_score']

        # Calculate averages
        top_avg = {metric: np.mean([agent[metric] for agent in top_5]) for metric in metrics}
        bottom_avg = {metric: np.mean([agent[metric] for agent in bottom_5]) for metric in metrics}

        # Create comparison chart
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(metrics))
        width = 0.35

        top_values = list(top_avg.values())
        bottom_values = list(bottom_avg.values())

        bars1 = ax.bar(x - width/2, top_values, width, label='Top 5', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x + width/2, bottom_values, width, label='Bottom 5', color='#DC143C', alpha=0.8)

        # Add value labels on bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                   f'{top_values[i]:.2f}', ha='center', va='bottom', fontsize=10)
            ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                   f'{bottom_values[i]:.2f}', ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Average Score')
        ax.set_title('Top 5 vs Bottom 5 Performers - Metric Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_score', '').replace('_', ' ').title() for m in metrics], 
                          rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"⚖️ Top vs bottom comparison saved to {save_path}")
        return save_path

    def create_weakness_analysis(self, bottom_analysis_data: List[Dict], save_path='visualizations/weakness_analysis.png'):
        """Analyze common weaknesses in bottom performers"""
        if not bottom_analysis_data:
            print("⚠️ No bottom analysis data available for weakness visualization")
            return None

        # Extract primary weaknesses
        weaknesses = {}
        for agent in bottom_analysis_data:
            if 'analysis' in agent and 'primary_weakness' in agent['analysis']:
                weakness = agent['analysis']['primary_weakness']
                weaknesses[weakness] = weaknesses.get(weakness, 0) + 1

        if not weaknesses:
            print("⚠️ No weakness data found")
            return None

        # Create weakness distribution chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Pie chart
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(weaknesses)))
        wedges, texts, autotexts = ax1.pie(weaknesses.values(), labels=weaknesses.keys(), 
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax1.set_title('Primary Weaknesses Distribution\n(Bottom 15 Performers)')

        # Bar chart
        bars = ax2.bar(range(len(weaknesses)), list(weaknesses.values()), 
                      color=colors_pie, alpha=0.8)
        ax2.set_xlabel('Primary Weakness')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Primary Weaknesses Count')
        ax2.set_xticks(range(len(weaknesses)))
        ax2.set_xticklabels(weaknesses.keys(), rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, count in zip(bars, weaknesses.values()):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"🎯 Weakness analysis saved to {save_path}")
        return save_path

    def create_score_trends(self, leaderboard_data: Dict, save_path='visualizations/score_trends.png'):
        """Create score trend visualization across all agents"""
        full_leaderboard = leaderboard_data['full_leaderboard']

        fig, ax = plt.subplots(figsize=(14, 8))

        ranks = [agent['rank'] for agent in full_leaderboard]
        scores = [agent['overall_score'] for agent in full_leaderboard]

        # Color-code by performance
        colors = []
        for score in scores:
            if score >= 90:
                colors.append('#2E8B57')  # Excellent
            elif score >= 75:
                colors.append('#4682B4')  # Good
            elif score >= 60:
                colors.append('#FF8C00')  # Fair
            else:
                colors.append('#DC143C')  # Poor

        scatter = ax.scatter(ranks, scores, c=colors, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

        # Add trend line
        z = np.polyfit(ranks, scores, 1)
        p = np.poly1d(z)
        ax.plot(ranks, p(ranks), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')

        ax.set_xlabel('Agent Rank')
        ax.set_ylabel('Overall Score')
        ax.set_title('Agent Performance Across All Ranks')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add performance thresholds
        ax.axhline(y=90, color='green', linestyle=':', alpha=0.7, label='Excellent Threshold')
        ax.axhline(y=75, color='blue', linestyle=':', alpha=0.7, label='Good Threshold')  
        ax.axhline(y=60, color='orange', linestyle=':', alpha=0.7, label='Fair Threshold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Score trends saved to {save_path}")
        return save_path

    def generate_all_visualizations(self, leaderboard_data: Dict, bottom_analysis_data: List[Dict] = None):
        """Generate all visualizations and return file paths"""
        print("🎨 Generating visualizations...")

        viz_files = {}

        # Core visualizations
        viz_files['performance_dist'] = self.create_performance_distribution()
        viz_files['correlation_heatmap'] = self.create_correlation_heatmap()
        viz_files['top_bottom_comparison'] = self.create_top_bottom_comparison(leaderboard_data)
        viz_files['score_trends'] = self.create_score_trends(leaderboard_data)

        # Optional weakness analysis (if AI analysis available)
        if bottom_analysis_data:
            viz_files['weakness_analysis'] = self.create_weakness_analysis(bottom_analysis_data)

        # Save visualization index
        viz_index = {
            'visualizations': viz_files,
            'timestamp': pd.Timestamp.now().isoformat(),
            'description': 'Agent evaluation visualizations'
        }

        with open('reports/visualizations_index.json', 'w') as f:
            json.dump(viz_index, f, indent=2)

        print(f"🎨 Generated {len(viz_files)} visualizations")
        return viz_files
