#!/usr/bin/env python3
"""
Three-Way Prompt Comparison Script

This script:
1. Loads the leaderboard CSV
2. Finds high, middle, and low-performing prompts based on accuracy
3. Creates a three-way comparison for better insight
4. Requests beginner-friendly analysis from Claude
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict

import anthropic  # Optional: for direct API calls
import pandas as pd

# File paths
LEADERBOARD_FILE = "leaderboard.csv"
VALID_PROMPTS_DIR = "valid_prompts"
OUTPUT_DIR = "prompt_analysis"
CLAUDE_API_KEY = ''  # Add API KEY

# Constants
PROMPT_TYPE_MAPPING = {
    "zero": "Zero-shot",
    "zero-shot": "Zero-shot",
    "zero shot": "Zero-shot",
    "few": "Few-shot",
    "few-shot": "Few-shot",
    "few shot": "Few-shot",
    "chain": "Chain-of-thought",
    "chain-of-thought": "Chain-of-thought",
    "chain of thought": "Chain-of-thought",
    "cot": "Chain-of-thought",
    "self": "Self-consistency",
    "self-consistency": "Self-consistency"
}
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"


def load_leaderboard_data():
    """Load the leaderboard data from CSV"""
    if not os.path.exists(LEADERBOARD_FILE):
        raise FileNotFoundError(f"Leaderboard file {LEADERBOARD_FILE} not found")

    # Read CSV into pandas DataFrame
    leaderboard_df = pd.read_csv(LEADERBOARD_FILE)

    # Convert to list of dictionaries
    leaderboard = leaderboard_df.to_dict('records')

    print(f"Loaded leaderboard with {len(leaderboard)} entries")
    return leaderboard


def load_valid_prompts():
    """Load all valid prompts from the directory"""
    if not os.path.exists(VALID_PROMPTS_DIR):
        raise FileNotFoundError(f"Valid prompts directory {VALID_PROMPTS_DIR} not found")

    prompt_files = [f for f in os.listdir(VALID_PROMPTS_DIR) if f.endswith('.json')]
    prompts = []

    for file in prompt_files:
        try:
            with open(os.path.join(VALID_PROMPTS_DIR, file), 'r') as f:
                prompt_data = json.load(f)

            # Add filename to the data
            prompt_data['filename'] = file
            prompts.append(prompt_data)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    print(f"Loaded {len(prompts)} valid prompts")
    return prompts


def normalize_prompt_type(prompt_type):
    """Normalize prompt type strings to standard categories"""
    if not prompt_type:
        return "Unknown"

    prompt_type_lower = prompt_type.lower()

    for key, value in PROMPT_TYPE_MAPPING.items():
        if key in prompt_type_lower:
            return value

    return prompt_type  # Return original if no match


def group_prompts_by_type(leaderboard_data):
    """Group prompts by their normalized type"""
    grouped = defaultdict(list)

    for entry in leaderboard_data:
        prompt_type = entry.get('prompt_type', 'Unknown')
        normalized_type = normalize_prompt_type(prompt_type)
        entry['normalized_type'] = normalized_type
        grouped[normalized_type].append(entry)

    # Sort each group by accuracy
    for prompt_type in grouped:
        grouped[prompt_type].sort(key=lambda x: float(x.get('accuracy', 0)), reverse=True)

    return grouped


def find_prompt_for_entry(entry, valid_prompts):
    """Find the complete prompt for a leaderboard entry by cross-referencing data"""
    submission_id = entry.get('submission_id')
    sender_name = entry.get('sender_name')
    prompt_num = entry.get('prompt_num')

    # First try to find by submission_id in valid_prompts
    for prompt in valid_prompts:
        if prompt.get('submission_id') == submission_id:
            return prompt

    # Next, try to find by sender and prompt number
    candidates = []
    for prompt in valid_prompts:
        if (prompt.get('sender_name') == sender_name and
                str(prompt.get('prompt_num')) == str(prompt_num)):
            candidates.append(prompt)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # If multiple matches, use content hash if available
        if 'content_hash' in entry and entry['content_hash']:
            content_hash = str(entry['content_hash'])
            for prompt in candidates:
                # Generate hash for comparison
                prompt_hash = hashlib.md5(prompt.get('prompt_text', '').encode()).hexdigest()
                if prompt_hash.startswith(content_hash) or content_hash.startswith(prompt_hash):
                    return prompt

    # If no exact match, just return the first candidate or None
    return candidates[0] if candidates else None


def select_prompts_for_comparison(grouped_prompts, args):
    """Select prompts for three-way comparison based on user input"""
    comparisons = []

    if args.high_rank and args.mid_rank and args.low_rank:
        # User specified exact ranks to compare
        for prompt_type, prompts in grouped_prompts.items():
            if args.prompt_type and prompt_type != args.prompt_type:
                continue

            if len(prompts) <= max(args.high_rank, args.mid_rank, args.low_rank):
                print(
                    f"Not enough prompts for {prompt_type} to compare ranks {args.high_rank}, {args.mid_rank}, and {args.low_rank}")
                continue

            high_entry = prompts[args.high_rank - 1]  # -1 because ranks are 1-based
            mid_entry = prompts[args.mid_rank - 1]
            low_entry = prompts[args.low_rank - 1]

            comparisons.append({
                'prompt_type': prompt_type,
                'high': high_entry,
                'mid': mid_entry,
                'low': low_entry
            })
    else:
        # Auto-select high, mid, and low performers
        for prompt_type, prompts in grouped_prompts.items():
            if args.prompt_type and prompt_type != args.prompt_type:
                continue

            if len(prompts) < 3:  # Need at least 3 for high, mid, low
                print(f"Not enough prompts ({len(prompts)}) for type {prompt_type} to make three-way comparison")
                continue

            # Get high performers (from the top)
            high_performers = prompts[:args.high]

            # Get middle performers (from the middle)
            mid_index = len(prompts) // 2
            mid_performers = prompts[mid_index:mid_index + args.mid]

            # Get low performers (from the bottom, but not absolute bottom)
            low_index = max(len(prompts) - args.low, 0)
            if low_index == 0:
                low_performers = prompts[-args.low:]
            else:
                # Get a few from toward the bottom, avoiding absolute worst
                low_performers = prompts[low_index - args.low:low_index]

            # Create combinations
            for high_entry in high_performers:
                for mid_entry in mid_performers:
                    for low_entry in low_performers:
                        comparisons.append({
                            'prompt_type': prompt_type,
                            'high': high_entry,
                            'mid': mid_entry,
                            'low': low_entry
                        })

    return comparisons


def process_entry_for_comparison(entry, valid_prompts):
    """Process a leaderboard entry for comparison"""
    prompt_data = find_prompt_for_entry(entry, valid_prompts)

    if not prompt_data:
        print(f"Could not find valid prompt for: {entry.get('sender_name')}, Prompt #{entry.get('prompt_num')}")
        return None

    processed_entry = {
        'entry': entry,
        'prompt_data': prompt_data
    }

    return processed_entry


def generate_analysis_prompt(comparison_data, valid_prompts, cross_type=False):
    """Generate a prompt for Claude to analyze the three-way comparison"""
    # Process entries
    high_processed = process_entry_for_comparison(comparison_data['high'], valid_prompts)
    mid_processed = process_entry_for_comparison(comparison_data['mid'], valid_prompts)
    low_processed = process_entry_for_comparison(comparison_data['low'], valid_prompts)

    if not high_processed or not mid_processed or not low_processed:
        print("Could not process all entries for comparison")
        return None

    prompt_type = comparison_data['prompt_type']
    high_data = high_processed
    mid_data = mid_processed
    low_data = low_processed

    # Format the prompt
    if cross_type:
        prompt = f"""
You're analyzing three prompts designed for sentiment classification that were evaluated on the same test dataset. 
These prompts are of different types: the high-performer is a {high_data['entry'].get('normalized_type')} prompt, 
the mid-performer is a {mid_data['entry'].get('normalized_type')} prompt, and the low-performer is a 
{low_data['entry'].get('normalized_type')} prompt.

In this special sentiment classification task, positive reviews are labeled "SAD" and negative reviews are labeled "HAPPY" 
(this is intentionally reversed from normal sentiment analysis).

I need you to analyze what makes each prompt more or less effective, focusing especially on concrete, 
practical lessons that a beginner to prompt engineering could understand and apply.

# High-Performing Prompt ({high_data['entry'].get('normalized_type')}, Accuracy: {high_data['entry'].get('accuracy')}%)
Rank: {high_data['entry'].get('rank', '?')}
Text:
```
{high_data['prompt_data'].get('prompt_text', 'Not available')}
```
"""
    else:
        prompt = f"""
You're analyzing three prompts of the same type ({prompt_type}) designed for sentiment classification.
They were all evaluated on the same test dataset but achieved different performance levels.

In this special sentiment classification task, positive reviews are labeled "SAD" and negative reviews are labeled "HAPPY" 
(this is intentionally reversed from normal sentiment analysis).

I need you to analyze what makes each prompt more or less effective, focusing especially on concrete, 
practical lessons that a beginner to prompt engineering could understand and apply.

# High-Performing Prompt (Accuracy: {high_data['entry'].get('accuracy')}%)
Rank: {high_data['entry'].get('rank', '?')}
Text:
```
{high_data['prompt_data'].get('prompt_text', 'Not available')}
```
"""

    prompt += f"""
# Mid-Performing Prompt (Accuracy: {mid_data['entry'].get('accuracy')}%)
Rank: {mid_data['entry'].get('rank', '?')}
Text:
```
{mid_data['prompt_data'].get('prompt_text', 'Not available')}
```

# Low-Performing Prompt (Accuracy: {low_data['entry'].get('accuracy')}%)
Rank: {low_data['entry'].get('rank', '?')}
Text:
```
{low_data['prompt_data'].get('prompt_text', 'Not available')}
```

# Analysis Task
Please analyze these three prompts to explain what makes them more or less effective for the sentiment classification task.

Your analysis should:
1. Be accessible to beginners in prompt engineering
2. Use clear, concrete examples from the prompts to illustrate your points
3. Highlight specific techniques that worked well in the high-performing prompt
4. Point out specific issues or missed opportunities in the lower-performing prompts
5. Provide 3-5 practical, actionable tips for writing better prompts based on this comparison

Focus on patterns and principles that would be useful for someone new to prompt engineering.
Explain WHY certain approaches work better, not just WHAT the differences are.
"""

    return prompt


def call_claude_api(prompt, api_key, model=CLAUDE_MODEL, max_tokens=4000):
    """Call Claude API to get the analysis"""
    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the text content from the message
        if hasattr(message, 'content') and isinstance(message.content, list):
            # New API returns content as a list of objects
            content_parts = []
            for block in message.content:
                # Handle different types of content blocks
                if hasattr(block, 'text'):
                    content_parts.append(block.text)
                elif hasattr(block, 'type') and block.type == 'text' and hasattr(block, 'text'):
                    content_parts.append(block.text)
                elif isinstance(block, dict) and block.get('type') == 'text':
                    content_parts.append(block.get('text', ''))
            return '\n'.join(content_parts)
        elif hasattr(message, 'content') and isinstance(message.content, str):
            # Direct string content
            return message.content
        else:
            # Fallback: try to access text directly from response
            if hasattr(message, 'text'):
                return message.text
            # Last resort: just convert the whole response to string
            return str(message)
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error traceback
        return None


def save_analysis_prompt(prompt, filename, output_dir=OUTPUT_DIR):
    """Save the analysis prompt to a file"""
    os.makedirs(output_dir, exist_ok=True)

    filepath = f"{output_dir}/{filename}"
    with open(filepath, 'w') as f:
        f.write(prompt)

    return filepath


def save_analysis_response(response, filename, output_dir=OUTPUT_DIR):
    """Save the analysis response to a file"""
    os.makedirs(output_dir, exist_ok=True)

    filepath = f"{output_dir}/{filename}"

    # Ensure response is a string
    if response is None:
        response_text = "No response received from API"
    elif not isinstance(response, str):
        try:
            response_text = str(response)
        except:
            response_text = "Could not convert response to string"
    else:
        response_text = response

    with open(filepath, 'w') as f:
        f.write(response_text)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="Compare high, mid, and low performing prompts")
    parser.add_argument("--high", type=int, default=1, help="Number of high performers to analyze per type")
    parser.add_argument("--mid", type=int, default=1, help="Number of middle performers to analyze per type")
    parser.add_argument("--low", type=int, default=1, help="Number of low performers to analyze per type")
    parser.add_argument("--high-rank", type=int, help="Specific rank of high performer to analyze")
    parser.add_argument("--mid-rank", type=int, help="Specific rank of mid performer to analyze")
    parser.add_argument("--low-rank", type=int, help="Specific rank of low performer to analyze")
    parser.add_argument("--api-key", type=str, default=CLAUDE_API_KEY, help="Claude API key for analysis")
    parser.add_argument("--save-only", action="store_true", help="Only save prompts without calling API")
    parser.add_argument("--prompt-type", type=str, help="Specific prompt type to analyze")
    parser.add_argument("--cross-type", action="store_true", help="Compare prompts across different types")
    args = parser.parse_args()

    # Load data
    leaderboard_data = load_leaderboard_data()
    valid_prompts = load_valid_prompts()

    # Group prompts by type
    grouped_prompts = group_prompts_by_type(leaderboard_data)

    # Print prompt types found in leaderboard
    print("\nPrompt types found in leaderboard:")
    for prompt_type, prompts in grouped_prompts.items():
        print(f"  {prompt_type}: {len(prompts)} prompts")

    # Select prompts for comparison
    comparisons = select_prompts_for_comparison(grouped_prompts, args)

    if not comparisons:
        print("No comparisons to analyze based on your criteria")
        return

    print(f"\nSelected {len(comparisons)} three-way comparisons to analyze")

    # Process each comparison
    for i, comparison in enumerate(comparisons, 1):
        print(f"\nProcessing comparison {i}/{len(comparisons)}: {comparison['prompt_type']}")
        print(
            f"  High performer: {comparison['high'].get('sender_name')} (Rank {comparison['high'].get('rank')}, Accuracy {comparison['high'].get('accuracy')}%)")
        print(
            f"  Mid performer: {comparison['mid'].get('sender_name')} (Rank {comparison['mid'].get('rank')}, Accuracy {comparison['mid'].get('accuracy')}%)")
        print(
            f"  Low performer: {comparison['low'].get('sender_name')} (Rank {comparison['low'].get('rank')}, Accuracy {comparison['low'].get('accuracy')}%)")

        # Generate analysis prompt
        analysis_prompt = generate_analysis_prompt(comparison, valid_prompts, args.cross_type)

        if not analysis_prompt:
            print("  Could not generate analysis prompt, skipping")
            continue

        # Generate filename
        filename_base = f"{comparison['prompt_type'].lower().replace('-', '_')}_rank{comparison['high'].get('rank')}_vs_rank{comparison['mid'].get('rank')}_vs_rank{comparison['low'].get('rank')}"

        # Save the prompt
        prompt_file = save_analysis_prompt(analysis_prompt, f"{filename_base}_prompt.txt")
        print(f"  Saved analysis prompt to {prompt_file}")

        # Call API if key is provided and not save-only
        if args.api_key and not args.save_only:
            print(f"  Calling Claude API for analysis...")
            response = call_claude_api(analysis_prompt, args.api_key)

            if response:
                response_file = save_analysis_response(response, f"{filename_base}_response.txt")
                print(f"  Saved analysis response to {response_file}")
            else:
                print(f"  ❌ Failed to get response from Claude API")

        # Add a sleep if making multiple API calls
        if args.api_key and not args.save_only and i < len(comparisons):
            time.sleep(1)  # Rate limiting

    print("\nAnalysis process complete!")

    if not args.api_key and not args.save_only:
        print("\nNote: No API key provided. To analyze with Claude API, run with --api-key YOUR_API_KEY")

    if args.save_only:
        print(f"\nPrompts saved to {OUTPUT_DIR}. You can now use these prompts with the Claude API manually.")


if __name__ == "__main__":
    main()
