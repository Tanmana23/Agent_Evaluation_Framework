import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import json
import re
from datetime import datetime

# Suppress the specific tqdm warning in Streamlit
st.write("""
    <style>
    .stProgress > div > div > div > div {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Dependency Imports ---
# Make sure to have these installed:
# pip install streamlit pandas numpy google-generativeai sentence-transformers transformers torch vaderSentiment matplotlib seaborn
try:
    import google.generativeai as genai
    from sentence_transformers import SentenceTransformer, util
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    st.error(f"A required library is not installed. Please run `pip install -r requirements.txt`. Error: {e}")
    st.stop()

# --- IMPORTS FROM PROJECT MODULES ---
# These files must be in the same directory as app.py
try:
    # Imports from 'metrics_for_main' directory
    from metrics_for_main.ai_judge import AIJudge
    from metrics_for_main.advanced_scorers import score_contradiction_hallucination, score_assumption
    from metrics_for_main.scorers import AdvancedInstructionFollowingScorer, score_coherence
    from metrics_for_main.novel_scoring_methods import score_cognitive_agility_v2, score_sentiment_risk, score_rhetoric_analysis
    
    # Imports from 'evaluators' directory
    from evaluators.streamlined_analyzer import StreamlinedAnalyzer
    from evaluators.focused_bottom_analyzer import FocusedBottomAnalyzer
    from evaluators.visualization_generator import VisualizationGenerator
    
    # Import from root directory
    from main import calculate_overall_score
except ImportError as e:
    st.error(f"Failed to import a project file. Make sure your directory structure matches the required layout (e.g., evaluators/, metrics_for_main/). Error: {e}")
    st.stop()


# --- STREAMLIT-OPTIMIZED HELPER FUNCTIONS ---

# =======================================================
# Model Caching (Streamlit-Optimized)
# =======================================================
@st.cache_resource
def get_model(model_name, model_class, tokenizer_class=None):
    """Robustly loads and caches any transformer model using Streamlit's caching."""
    st.info(f"Loading '{model_name}' model for the first time... (This may take a moment)")
    if tokenizer_class:
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        return {"model": model, "tokenizer": tokenizer}
    else:
        model = model_class(model_name)
        return model

@st.cache_resource
def get_embedding_model(model_name):
    """Efficiently loads and caches an embedding model."""
    st.info(f"Loading embedding model '{model_name}' for the first time...")
    return SentenceTransformer(model_name)

@st.cache_resource
def get_sentiment_analyzer():
    """Initializes and returns a single instance of the sentiment analyzer."""
    return SentimentIntensityAnalyzer()

# =======================================================
# MAIN PIPELINE LOGIC
# =======================================================
def run_evaluation_pipeline(df, api_key):
    """The full, orchestrated pipeline using imported modules."""
    st.write("--- Running Tier 1: High-Speed Automated Scoring ---")
    progress_bar = st.progress(0, text="Initializing...")
    
    instruction_scorer = AdvancedInstructionFollowingScorer()
    total_steps = 7
    
    df['instruction_following_score'], df['instruction_following_explanation'] = zip(*df.apply(lambda row: instruction_scorer.score(str(row.get('prompt_text','')), str(row.get('response_text','')), str(row.get('agent_persona',''))), axis=1))
    progress_bar.progress(1/total_steps, text="Instruction Following Scored...")

    df['coherence_score'] = df.apply(lambda row: score_coherence(str(row.get('prompt_text','')), str(row.get('response_text',''))), axis=1)
    progress_bar.progress(2/total_steps, text="Coherence Scored...")

    df['cognitive_agility_score'] = df.apply(lambda row: score_cognitive_agility_v2(str(row.get('response_text','')), str(row.get('ground_truth',''))), axis=1)
    df['sentiment_risk_score'] = df.apply(lambda row: score_sentiment_risk(str(row.get('response_text',''))), axis=1)
    df['rhetoric_analysis_score'] = df.apply(lambda row: score_rhetoric_analysis(str(row.get('response_text',''))), axis=1)
    progress_bar.progress(4/total_steps, text="Novel Metrics Scored...")

    df['hallucination_score'] = df.apply(lambda row: score_contradiction_hallucination(str(row.get('response_text','')), str(row.get('ground_truth',''))), axis=1)
    progress_bar.progress(5/total_steps, text="Hallucination Scored...")
    
    df['assumption_score'] = df.apply(lambda row: score_assumption(str(row.get('response_text',''))), axis=1)
    progress_bar.progress(6/total_steps, text="Assumption Scored...")

    st.success("Tier 1 Scoring Complete.")
    progress_bar.progress(7/total_steps, text="Tier 1 Complete.")
    
    st.write("--- Running Tier 2: Intelligent Escalation to AI Judge ---")
    suspicion_trigger = ((df['hallucination_score'] < 0.75) | (df['rhetoric_analysis_score'] >= 2) | (df['assumption_score'] < 0.6) | (df['instruction_following_score'] < 0.5) | (df['coherence_score'] < 0.35))
    responses_to_judge = df[suspicion_trigger]
    
    df['ai_judge_score'], df['ai_judge_explanation'] = np.nan, ""
    
    if not responses_to_judge.empty and api_key:
        st.info(f"Identified {len(responses_to_judge)} suspicious responses for review by AI Judge.")
        judge = AIJudge(api_key=api_key)
        
        judge_progress = st.progress(0, text="AI Judge Review Starting...")
        total_to_judge = len(responses_to_judge)
        request_counter = 0

        for i, (index, row) in enumerate(responses_to_judge.iterrows()):
            # CORRECTED RATE LIMITING LOGIC
            if request_counter > 0 and request_counter % 15 == 0:
                with st.spinner("Processing"):
                    time.sleep(61)

            score, explanation = judge.score_response_holistically(str(row['response_text']), str(row['ground_truth']))
            df.loc[index, 'ai_judge_score'], df.loc[index, 'ai_judge_explanation'] = score, explanation
            request_counter += 1
            judge_progress.progress((i + 1) / total_to_judge, text=f"AI Judge Reviewed {i+1}/{total_to_judge} responses...")
            
        st.success("AI Judge Review Complete.")
    else:
        st.warning("Skipping AI Judge. No suspicious responses found or API key not provided.")
        
    st.write("--- Calculating Final Scores ---")
    df['overall_score'] = df.apply(calculate_overall_score, axis=1)
    st.success("Evaluation pipeline finished successfully!")
    return df

# =======================================================
# STREAMLIT UI APPLICATION
# =======================================================
st.set_page_config(layout="wide", page_title="Agentic Evaluation Platform")

st.title("Agentic Evaluation Platform")
st.markdown("Upload your agent interaction data to run a comprehensive, multi-layered evaluation pipeline.")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("1. Upload Raw Data (CSV)", type="csv")
    api_key = st.text_input("2. Enter Gemini API Key (Required for AI Judge & Analysis)", type="password")
    run_button = st.button("Run Full Evaluation", type="primary", use_container_width=True)

if run_button:
    if uploaded_file is not None and api_key:
        st.session_state.clear()
        try:
            # Create directories if they don't exist
            os.makedirs("visualizations", exist_ok=True)
            os.makedirs("reports", exist_ok=True)

            raw_df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = raw_df
            
            with st.status("Starting Evaluation Pipeline...", expanded=True) as status:
                st.session_state.scored_df = run_evaluation_pipeline(raw_df.copy(), api_key)
                status.update(label="Scoring Complete! Now running analysis...", state="running")
                
                # The original StreamlinedAnalyzer expects a file path, so we save the scored df to a temporary file
                temp_csv_path = "reports/temp_scored_data.csv"
                st.session_state.scored_df.to_csv(temp_csv_path, index=False)
                
                analyzer = StreamlinedAnalyzer(temp_csv_path)
                st.session_state.summary = analyzer.generate_summary()
                st.session_state.leaderboard = analyzer.generate_leaderboard()
                
                bottom_15_data = st.session_state.leaderboard['bottom_15']
                if bottom_15_data:
                    enhanced_bottom_data = analyzer.get_bottom_15_for_ai_analysis(bottom_15_data)
                    ai_analyzer = FocusedBottomAnalyzer(api_key)
                    # CORRECTED: Calling the main method which contains its own rate limiting
                    st.session_state.bottom_analysis = ai_analyzer.analyze_bottom_performers(enhanced_bottom_data)
                else:
                    st.session_state.bottom_analysis = None
                
                status.update(label="📊 Generating Visualizations...", state="running")
                viz_gen = VisualizationGenerator(st.session_state.scored_df)
                st.session_state.viz_files = viz_gen.generate_all_visualizations(
                    st.session_state.leaderboard, 
                    st.session_state.bottom_analysis
                )
                
                status.update(label="🎉 Evaluation Complete!", state="complete")

            st.session_state.run_complete = True
        except Exception as e:
            st.error(f"An error occurred during the pipeline: {e}")
            st.exception(e)
            st.session_state.run_complete = False

    else:
        st.warning("Please upload a CSV file and enter your Gemini API key.")

if st.session_state.get('run_complete', False):
    st.header("Evaluation Results")
    
    summary = st.session_state.summary
    leaderboard = st.session_state.leaderboard
    bottom_analysis = st.session_state.bottom_analysis
    viz_files = st.session_state.viz_files
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Summary", "🏆 Leaderboard", "🎨 Performance Visuals", "🧠 Weakness Analysis", "📋 Full Data"])
    
    with tab1:
        st.subheader("Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Unique Agents", summary['dataset_info']['unique_agents'])
        col2.metric("Average Score", f"{summary['performance_stats']['avg_score']:.2f}")
        col3.metric("Median Score", f"{summary['performance_stats']['median_score']:.2f}")
        col4.metric("Top Score", f"{summary['performance_stats']['max_score']:.2f}")
        
        if 'performance_dist' in viz_files and os.path.exists(viz_files['performance_dist']):
            st.image(viz_files['performance_dist'])

    with tab2:
        st.subheader("Top 15 Performing Agents")
        st.dataframe(pd.DataFrame(leaderboard['top_15']), use_container_width=True)
        
        st.subheader("Bottom 15 Performing Agents")
        st.dataframe(pd.DataFrame(leaderboard['bottom_15']), use_container_width=True)
        
    with tab3:
        st.subheader("Performance Visualizations")
        if 'top_bottom_comparison' in viz_files and os.path.exists(viz_files['top_bottom_comparison']):
            st.image(viz_files['top_bottom_comparison'])
        if 'correlation_heatmap' in viz_files and os.path.exists(viz_files['correlation_heatmap']):
            st.image(viz_files['correlation_heatmap'])
        else:
            st.warning("Not enough data to generate a correlation heatmap.")

    with tab4:
        st.subheader("AI-Powered Weakness Analysis (Bottom 15)")
        if 'weakness_analysis' in viz_files and viz_files['weakness_analysis'] and os.path.exists(viz_files['weakness_analysis']):
            st.image(viz_files['weakness_analysis'])
        
        if bottom_analysis:
            for analysis in bottom_analysis:
                with st.expander(f"**Agent:** {analysis['agent_id']} | **Rank:** {analysis['rank']} | **Score:** {analysis['overall_score']:.2f}"):
                    res = analysis['analysis']
                    st.markdown(f"**Primary Weakness:** `{res.get('primary_weakness', 'N/A')}`")
                    st.markdown(f"**Main Issue:** {res.get('main_issue', 'N/A')}")
                    st.markdown(f"**Improvement Potential:** {res.get('improvement_potential', 'N/A')}")
                    st.markdown("**Recommendations:**")
                    for rec in res.get('recommendations', []):
                        st.markdown(f"- {rec}")
        else:
            st.info("No data available for weakness analysis.")
            
    with tab5:
        st.subheader("Full Scored Dataset")
        st.dataframe(st.session_state.scored_df)
        
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(st.session_state.scored_df)
        st.download_button(
            label="Download Scored Data as CSV",
            data=csv,
            file_name='scored_agent_data.csv',
            mime='text/csv',
        )

