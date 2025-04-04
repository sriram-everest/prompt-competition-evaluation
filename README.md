# Prompt Engineering Homework - Evaluation Guide

This guide helps you set up and run the evaluation system for the Amazon Review Sentiment Classification Challenge
submissions.

## Prerequisites

1. Python 3.10+ installed
2. Pip package manager
3. Sufficient disk space (~5GB) for model download
4. Ollama(https://ollama.com/) with 'gemma2:2b' downloaded - `ollama pull gemma2:2b`
5. Create a new sample dataset with HuggingFace (requires HF_KEY environment variable from your HuggingFace account) -
   using `sampleset_creator.py`.

## Setup Instructions

1. **Create virtual environment**

```bash
   python -m venv venv
   source venv/bin/activate
```

2. **Install required packages**:
   ```bash
   pip install -r requirements
   ```

3. **Configure email access**:
    - Edit the script to add your Gmail credentials
    - For Gmail, you'll need to create an App Password:
      (Google Account → Security → App Passwords)
    - Update `EMAIL_ADDRESS` and `EMAIL_PASSWORD` in the script
    - The script is configured to search for emails with the label "prompthomework"

4. **Configure CLAUDE_API_KEY** - needed for `extract_prompts.py` as well as `comparison_script.py`.

## Running the Evaluation

### Download emails

```bash
python email_downloader.py
```

### Extract prompts from downloaded emails

```bash
python extract_prompts.py
```

### Evaluate extracted prompts against Ollama running gemma 2 2b

```bash
python prompt_evaluator.py
```

### Compare high, mid and low scoring prompts by prompt type

```bash
python comparison_script.py
```

This will generate a prompt analysis of what makes a high performing prompt better than a mid, better than a low, using
Claude (Anthropic).

## Generated Files

The script maintains several files:

1. **evaluation_state.json**: Tracks which emails have been processed
2. **evaluation_results.json**: Complete evaluation results including all predictions
3. **leaderboard.json**: Current rankings in JSON format
4. **leaderboard.csv**: Current rankings in CSV format (for easy viewing)
5. **evaluation_summary.csv**: Summary of accuracy for each submission

## Result Interpretation

The accuracy shows how well each prompt performed at the "twisted" classification task:

- "SAD" = Positive sentiment (customer liked the product)
- "HAPPY" = Negative sentiment (customer disliked the product)

A higher accuracy indicates that the prompt was more successful at correctly applying this reversed labeling scheme.
