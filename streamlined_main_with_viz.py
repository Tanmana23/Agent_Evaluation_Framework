#!/usr/bin/env python3
"""
Streamlined AI Agent Evaluation - Phase 1 + Visualizations
Focused analytics with separate JSON outputs and visualizations for Streamlit integration

Usage: python streamlined_main_with_viz.py <csv_path> [gemini_api_key]
"""

import sys
import json
from datetime import datetime

# Import streamlined modules
from streamlined_analyzer import StreamlinedAnalyzer
from focused_bottom_analyzer import FocusedBottomAnalyzer  
from correlation_analyzer import CorrelationAnalyzer
from visualization_generator import VisualizationGenerator

class StreamlinedReportGeneratorWithViz:
    """Main orchestrator for focused evaluation reports with visualizations"""

    def __init__(self, csv_path: str, gemini_api_key: str = None):
        print("🚀 Initializing Streamlined Report Generator with Visualizations...")

        # Core analyzer
        self.analyzer = StreamlinedAnalyzer(csv_path)

        # Optional AI analyzer for bottom 15
        self.ai_analyzer = None
        if gemini_api_key and gemini_api_key.strip():
            try:
                self.ai_analyzer = FocusedBottomAnalyzer(gemini_api_key)
            except Exception as e:
                print(f"⚠️ AI analysis unavailable: {e}")

        # Correlation analyzer
        self.corr_analyzer = CorrelationAnalyzer(self.analyzer.df)

        # Visualization generator
        self.viz_generator = VisualizationGenerator(self.analyzer.df)

    def generate_all_reports(self):
        """Generate all report components as separate files + visualizations"""
        print("\n" + "="*60)
        print("📊 GENERATING STREAMLINED REPORTS + VISUALIZATIONS")
        print("="*60)

        # 1. Executive Summary
        print("\n📋 1/5: Executive Summary...")
        summary_file = self.analyzer.save_summary("executive_summary.json")

        # 2. Leaderboard (Top & Bottom 15)
        print("\n🏆 2/5: Agent Leaderboard...")
        leaderboard_file = self.analyzer.save_leaderboard("leaderboard.json")
        leaderboard_data = self.analyzer.generate_leaderboard()

        # 3. Correlations
        print("\n🔗 3/5: Metric Correlations...")
        correlation_file = self.corr_analyzer.save_correlations("correlations.json")
        heatmap_file = self.corr_analyzer.create_performance_heatmap("performance_heatmap.png")

        # 4. Bottom 15 AI Analysis (if available)
        analyzed_bottom = None
        if self.ai_analyzer:
            print("\n🤖 4/5: AI Analysis of Bottom 15...")
            bottom_15_enhanced = self.analyzer.get_bottom_15_for_ai_analysis(
                leaderboard_data['bottom_15']
            )
            analyzed_bottom = self.ai_analyzer.analyze_bottom_performers(bottom_15_enhanced)
            analysis_file = self.ai_analyzer.save_analysis(analyzed_bottom, "bottom_15_analysis.json")
        else:
            print("\n⏭️ 4/5: Skipped AI Analysis (no API key)")
            analysis_file = self._save_placeholder_analysis()

        # 5. Generate Visualizations
        print("\n🎨 5/5: Creating Visualizations...")
        viz_files = self.viz_generator.generate_all_visualizations(
            leaderboard_data, 
            analyzed_bottom
        )
        viz_files['performance_heatmap'] = heatmap_file
        # Generate index file
        index_file = self._save_index_file([summary_file, leaderboard_file, correlation_file, analysis_file], viz_files)

        return {
            'summary_file': summary_file,
            'leaderboard_file': leaderboard_file, 
            'correlation_file': correlation_file,
            'analysis_file': analysis_file,
            'visualization_files': viz_files,
            'index_file': index_file
        }

    def _save_placeholder_analysis(self):
        """Save placeholder when AI analysis is unavailable"""
        placeholder = {
            'bottom_15_analysis': [],
            'message': 'AI analysis unavailable - no Gemini API key provided',
            'timestamp': datetime.now().isoformat()
        }
        filename = "bottom_15_analysis.json"
        with open(filename, 'w') as f:
            json.dump(placeholder, f, indent=2)
        print(f"💾 Placeholder analysis saved to {filename}")
        return filename

    def _save_index_file(self, file_list, viz_files):
        """Save index file with all generated reports and visualizations"""
        index = {
            'report_metadata': {
                'generation_time': datetime.now().isoformat(),
                'version': '2.1-streamlined-with-viz',
                'description': 'Focused evaluation reports with visualizations'
            },
            'files': {
                'executive_summary': file_list[0],
                'leaderboard': file_list[1], 
                'correlations': file_list[2],
                'bottom_15_analysis': file_list[3]
            },
            'visualizations': viz_files,
            'streamlit_ready': True
        }

        filename = "report_index.json"
        with open(filename, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"💾 Report index saved to {filename}")
        return filename

    def display_summary(self):
        """Display a quick summary of results"""
        summary = self.analyzer.generate_summary()
        leaderboard = self.analyzer.generate_leaderboard()

        print("\n" + "="*60)
        print("📊 STREAMLINED EVALUATION SUMMARY")
        print("="*60)

        # Dataset info
        print(f"\n📈 Dataset Overview:")
        print(f"   • Total Agents: {summary['dataset_info']['total_agents']}")
        print(f"   • Average Score: {summary['performance_stats']['avg_score']:.1f}/100")

        # Performance distribution
        dist = summary['score_distribution']
        print(f"\n🎯 Performance Distribution:")
        print(f"   • Excellent (90+): {dist['excellent_90_plus']}")
        print(f"   • Good (75-89): {dist['good_75_89']}")
        print(f"   • Fair (60-74): {dist['fair_60_74']}")  
        print(f"   • Poor (<60): {dist['poor_below_60']}")

        # Top 3 and Bottom 3
        print(f"\n🏆 Top 3 Performers:")
        for agent in leaderboard['top_15'][:3]:
            print(f"   {agent['rank']}. {agent['agent_id']}: {agent['overall_score']:.1f}")

        print(f"\n📉 Bottom 3 Performers:")
        for agent in leaderboard['bottom_15'][-3:]:
            print(f"   {agent['rank']}. {agent['agent_id']}: {agent['overall_score']:.1f}")

        # AI analysis info
        if self.ai_analyzer:
            print(f"\n🤖 AI Analysis: ✅ Generated for bottom 15 performers")
        else:
            print(f"\n🤖 AI Analysis: ❌ Skipped (no API key)")

        print(f"\n🎨 Visualizations: ✅ Performance charts generated")

def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python streamlined_main_with_viz.py <csv_path> [gemini_api_key]")
        print("Example: python streamlined_main_with_viz.py data.csv your_api_key")
        sys.exit(1)

    csv_path = sys.argv[1]
    gemini_api_key = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        # Generate reports
        generator = StreamlinedReportGeneratorWithViz(csv_path, gemini_api_key)
        files = generator.generate_all_reports()

        # Display summary
        generator.display_summary()

        print("\n" + "="*60)
        print("✅ STREAMLINED REPORTS + VISUALIZATIONS COMPLETE!")
        print("="*60)
        print(f"📁 Files Generated:")
        for key, filename in files.items():
            if key == 'visualization_files':
                print(f"   📊 Visualizations:")
                for viz_key, viz_file in filename.items():
                    if viz_file:  # Only show non-None files
                        print(f"      • {viz_file}")
            else:
                print(f"   • {filename}")
        print("\n🚀 Ready for Streamlit integration!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
