# Agent Evaluation Framework

An automated, large-scale framework for the comprehensive evaluation of AI agents using hybrid scoring methods, intelligent AI judge escalation, and an interactive visualization dashboard.

## Overview

This framework provides an end-to-end pipeline for evaluating AI agents through:

- **High-Fidelity Synthetic Dataset Generation**: Create large, diverse datasets using multiple agent personas
- **Multi-Metric Hybrid Scoring**: A robust pipeline combining rule-based, ML, and heuristic methods
- **Intelligent Hybrid AI Judge Escalation**: Send complex cases requiring expert evaluation to a Gemini-powered AI Judge
- **Automated Insights & Reporting**: Generate in-depth reports, including AI-powered root-cause analysis for failing agents
- **Interactive Streamlit Dashboard**: A user-friendly interface for real-time evaluation and data exploration

## Architecture

The framework operates in several distinct phases:

### Phase 1: Dataset Generation
Sources prompts from the TruthfulQA dataset for a factual baseline and generates responses using 5 distinct agent personas (`factual`, `verbose`, `hallucinator`, `assumption_maker`, `non_follower`) to create a challenging testbed.

### Phase 2: Multi-Metric Scoring Pipeline
Evaluates responses across multiple dimensions:

- **Instruction Following**: Advanced parsing and compliance checking
- **Coherence**: Semantic similarity using sentence transformers
- **Hallucination Detection**: NLI-based contradiction analysis
- **Assumption Control**: Hybrid regex + semantic detection
- **Novel Heuristics**: Cognitive agility, sentiment risk, and rhetoric analysis

### Phase 3: AI Judge Integration
Escalates suspicious responses (e.g., those with low scores or high assumption rates) to a Gemini-powered AI Judge, which provides a structured holistic evaluation with explanations.

### Phase 4: Automated Analysis & Visualization
Generates executive summaries, performance leaderboards, and a comprehensive suite of visualizations (distributions, correlations, heatmaps). Includes an AI-powered root-cause analysis for the bottom-performing agents.

## Quick Start

### Prerequisites
- Python 3.8+
- A Google Gemini API Key (Required for AI Judge and AI Analysis functionality)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/samarth1964/agent_evaluation_framework.git
   cd agent_evaluation_framework
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up directories:**
   ```bash
   mkdir -p data reports visualizations
   ```

## Input Data Format

Before running an evaluation, ensure your input CSV data contains these required columns:

**Agent identifier** using one of the following columns:
- `agent_id`: A unique identifier for each individual agent
- `agent_persona`: A category or type for a group of agents

| Column | Description |
|--------|-------------|
| `prompt_text` | The input question/prompt given to the agent |
| `response_text` | The agent's generated response to be evaluated |
| `ground_truth` | The correct or expected answer for comparison |


> **Note**: The framework automatically detects your column structure. If both `agent_id` and `agent_persona` are present, `agent_id` will be prioritized for grouping. If neither is present, each row will be treated as a separate, anonymous agent.

## Workflows

You can either evaluate an existing dataset or generate a new one.

### Evaluating Agent Performance

You have two ways to run an evaluation on your dataset:

#### Option 1: Interactive Web Dashboard (Recommended)

Launch the Streamlit app for a full user interface:

```bash
streamlit run app.py
```

In the app:
1. Upload your CSV data
2. Enter your Gemini API key
3. Click "Run Full Evaluation" for a complete analysis with visualizations

#### Option 2: Command-Line Batch Processing

Run the entire pipeline for a given dataset and save all reports to the `reports/` and `visualizations/` folders:

```bash
python streamlined_main_with_viz.py your_data.csv YOUR_GEMINI_API_KEY
```

### Generating a Synthetic Dataset

If you want to create your own evaluation dataset from scratch, follow these steps:

```bash
# 1. Navigate to the data generation directory
cd data_generation

# 2. Prepare the prompts from TruthfulQA
python prompt_api.py

# 3. Generate agent responses (requires setting your API key inside the script)
python generate_data_api.py
```
## Customization & Extensibility

The framework is designed to be highly modular, allowing for easy customization.

### Scoring Weights

You can customize the importance of each metric by modifying the weights in the `calculate_overall_score()` function within `main.py`.

**Default Weights:**

```python
# Located in main.py
weights = {
    'hallucination': 0.35,           # Factual accuracy (most important)
    'instruction_following': 0.25,
    'assumption': 0.15,
    'coherence': 0.10,
    'cognitive_agility': 0.05,
    'sentiment_risk': 0.05,
    'rhetoric_analysis': 0.05
}
```

### AI Judge Triggers

You can change the criteria for when a response is sent to the AI Judge for a more rigorous (but more costly) evaluation. These thresholds are located in the `run_evaluation_pipeline` function in `app.py`.

**Default Triggers:**

```python
# Located in app.py
suspicion_trigger = (
    (df['hallucination_score'] < 0.75) | 
    (df['rhetoric_analysis_score'] >= 2) | 
    (df['assumption_score'] < 0.6) | 
    (df['instruction_following_score'] < 0.5) | 
    (df['coherence_score'] < 0.35)
)
```