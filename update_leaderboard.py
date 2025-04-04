#!/usr/bin/env python3
"""
Leaderboard Updater

This script scans the feedback directory for all feedback files,
identifies any participants missing from the leaderboard,
and updates the leaderboard accordingly.
"""

import os
import json
import re
import glob
import pandas as pd

# File paths
FEEDBACK_DIR = "feedback"
RESULTS_FILE = "evaluation_results.json"
LEADERBOARD_FILE = "leaderboard.json"

def update_leaderboard():
    print("Updating leaderboard with any missing entries from feedback files...")

    # Load existing results and leaderboard
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} evaluation results")
    else:
        print(f"Error: Results file {RESULTS_FILE} not found")
        return

    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, 'r') as f:
            leaderboard = json.load(f)
        print(f"Loaded leaderboard with {len(leaderboard)} entries")
    else:
        leaderboard = []
        print("No existing leaderboard found, creating new one")

    # Scan feedback directory
    if not os.path.exists(FEEDBACK_DIR):
        print(f"Error: Feedback directory {FEEDBACK_DIR} not found")
        return

    # Find all feedback files
    feedback_files = glob.glob(f"{FEEDBACK_DIR}/*.txt")
    print(f"Found {len(feedback_files)} feedback files")

    # Extract sender names and prompt numbers from filenames
    # Pattern: sender_name_promptN_model_dataset_submission.txt
    sender_entries = {}
    for feedback_file in feedback_files:
        filename = os.path.basename(feedback_file)

        # Try to extract sender name and prompt number
        match = re.match(r'([^_]+)_prompt(\d+)_', filename)
        if match:
            sender_name = match.group(1)
            prompt_num = match.group(2)

            # Get accuracy from feedback file
            accuracy = None
            try:
                with open(feedback_file, 'r') as f:
                    content = f.read()
                    acc_match = re.search(r'Accuracy:\s*([\d.]+)%', content)
                    if acc_match:
                        accuracy = float(acc_match.group(1))
            except Exception as e:
                print(f"Error reading {feedback_file}: {e}")

            if accuracy is not None:
                key = sender_name
                if key not in sender_entries or accuracy > sender_entries[key]['accuracy']:
                    sender_entries[key] = {
                        'sender_name': sender_name,
                        'prompt_num': prompt_num,
                        'accuracy': accuracy,
                        'feedback_file': feedback_file
                    }

    print(f"Extracted {len(sender_entries)} unique senders from feedback files")

    # Check which senders are missing from leaderboard
    leaderboard_senders = {entry['sender_name'] for entry in leaderboard}
    missing_senders = set(sender_entries.keys()) - leaderboard_senders

    if not missing_senders:
        print("No missing senders found in the leaderboard!")
        return

    print(f"Found {len(missing_senders)} senders missing from leaderboard:")
    for sender in missing_senders:
        print(f"  - {sender} (Accuracy: {sender_entries[sender]['accuracy']}%)")

    # Update results file with missing entries
    for sender in missing_senders:
        entry = sender_entries[sender]

        # Find matching result in results file
        matching_result = None
        for result in results:
            if (result.get('sender_name') == entry['sender_name'] and
                    str(result.get('prompt_num')) == entry['prompt_num']):
                matching_result = result
                break

        # If no matching result, create a placeholder
        if not matching_result:
            matching_result = {
                'submission_id': f"reconstructed_{sender}_{entry['prompt_num']}",
                'sender_name': entry['sender_name'],
                'prompt_num': entry['prompt_num'],
                'prompt_type': 'Unknown',
                'accuracy': entry['accuracy'],
                'evaluated_at': '2025-04-03T00:00:00',
                'model_name': 'sentiment-classifier',
                'using_ollama': True,
                'sample_size': 200,
                'dataset': 'fixed',
                'feedback_file': entry['feedback_file']
            }
            results.append(matching_result)
            print(f"Added reconstructed result for {sender}")

        # Add to leaderboard
        leaderboard_entry = {
            'rank': 0,  # Will be updated later
            'submission_id': matching_result.get('submission_id', f"reconstructed_{sender}_{entry['prompt_num']}"),
            'sender_name': entry['sender_name'],
            'prompt_num': entry['prompt_num'],
            'prompt_type': matching_result.get('prompt_type', 'Unknown'),
            'accuracy': entry['accuracy'],
            'evaluated_at': matching_result.get('evaluated_at', '2025-04-03T00:00:00'),
            'model_name': matching_result.get('model_name', 'sentiment-classifier'),
            'using_ollama': matching_result.get('using_ollama', True),
            'sample_size': matching_result.get('sample_size', 200),
            'dataset': matching_result.get('dataset', 'fixed')
        }

        leaderboard.append(leaderboard_entry)
        print(f"Added {sender} to leaderboard with accuracy {entry['accuracy']}%")

    # Sort and update ranks
    leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)
    for i, entry in enumerate(leaderboard, 1):
        entry['rank'] = i

    # Save updated results and leaderboard
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    # Also save as CSV
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv("leaderboard.csv", index=False)

    print(f"\nLeaderboard updated successfully!")
    print(f"Added {len(missing_senders)} missing senders")
    print(f"New leaderboard has {len(leaderboard)} entries")
    print(f"Updated files: {RESULTS_FILE}, {LEADERBOARD_FILE}, and leaderboard.csv")

if __name__ == "__main__":
    update_leaderboard()