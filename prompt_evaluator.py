import argparse
import asyncio
import datetime
import glob
import hashlib
import json
import os
import pickle
import re

import aiohttp
import pandas as pd
import requests
from tqdm import tqdm

# Ollama Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma2:2b"

# File paths
VALID_PROMPTS_DIR = "valid_prompts"
FEEDBACK_DIR = "feedback"
RESULTS_FILE = "evaluation_results.json"
LEADERBOARD_FILE = "leaderboard.json"
FIXED_SAMPLE_FILE = "fixed_evaluation_sample.pkl"

# Optimization settings
SAVE_FREQUENCY = 10
BATCH_SIZE = 32


class PromptEvaluator:
    def __init__(self, model_name=MODEL_NAME, ollama_url=OLLAMA_API_URL):
        """Initialize the evaluator with model and fixed dataset"""
        # Load state and results if they exist
        self.load_state_and_results()

        self.model_name = model_name
        self.ollama_url = ollama_url

        # Check if Ollama is running with the specified model
        print(f"Checking model availability: {model_name}")
        try:
            # Simple test call to verify connection and model
            payload = {
                "model": model_name,
                "prompt": "Hello, please respond with a single word.",
                "stream": False,
            }
            response = requests.post(ollama_url, json=payload)
            if response.status_code == 200:
                print(f"✅ Successfully connected to Ollama with model: {model_name}")
            else:
                print(f"⚠️ Error with Ollama model: {response.status_code} - {response.text}")
                print("Make sure Ollama is running and the model is available.")
                print(f"You can pull the model using: ollama pull {model_name}")
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to Ollama API: {str(e)}")
            print("Make sure Ollama is running at: " + ollama_url)
            raise

        # Load the fixed dataset
        self.test_data = self.load_fixed_sample()
        print(f"Loaded fixed dataset with {len(self.test_data)} examples")

        # Initialize results storage if not already loaded
        if not hasattr(self, 'results'):
            self.results = []

        # Track pending results
        self.pending_results = []

        # Create an in-memory prediction cache
        self.prediction_cache = {}

        # Create a content hash cache for prompts
        self.content_hash_cache = {}

    async def check_remaining_evaluations(self, directory=VALID_PROMPTS_DIR):
        """
        Check how many prompts remain to be evaluated and show detailed information
        """
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found")
            return 0, []

        # Load all prompt files
        prompt_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if not prompt_files:
            print(f"No prompt files found in {directory}")
            return 0, []

        print(f"Found {len(prompt_files)} total prompt files in {directory}")

        # Track stats
        to_evaluate = []
        already_evaluated = []
        skipped_no_placeholder = []

        # Group prompts by sender for better reporting
        by_sender = {}

        for prompt_file in prompt_files:
            try:
                # Load the prompt from the file
                with open(os.path.join(directory, prompt_file), 'r') as f:
                    prompt_data = json.load(f)

                # Extract prompt details
                sender_name = prompt_data.get('sender_name', 'Unknown')
                prompt_type = prompt_data.get('prompt_type', 'Unknown')
                prompt_text = prompt_data.get('prompt_text', '')
                prompt_num = prompt_data.get('prompt_num', '1')

                # Generate a submission ID based on the filename if not present
                submission_id = prompt_data.get('submission_id',
                                                hashlib.md5(prompt_file.encode()).hexdigest()[:20])

                # Skip prompts without proper placeholder
                if "%%REVIEW%%" not in prompt_text:
                    skipped_no_placeholder.append({
                        'file': prompt_file,
                        'sender_name': sender_name,
                        'prompt_num': prompt_num
                    })
                    continue

                # Generate content hash
                content_hash = self.generate_content_hash(prompt_text)

                # Check if feedback already exists
                feedback_exists, feedback_file = self.check_feedback_exists(
                    sender_name, prompt_num, submission_id, self.model_name, prompt_text
                )

                prompt_info = {
                    'file': prompt_file,
                    'sender_name': sender_name,
                    'prompt_num': prompt_num,
                    'prompt_type': prompt_type,
                    'submission_id': submission_id,
                    'content_hash': content_hash,
                    'feedback_file': feedback_file if feedback_exists else None
                }

                # Add to sender group
                if sender_name not in by_sender:
                    by_sender[sender_name] = {
                        'total': 0,
                        'to_evaluate': 0,
                        'evaluated': 0
                    }

                by_sender[sender_name]['total'] += 1

                if feedback_exists:
                    already_evaluated.append(prompt_info)
                    by_sender[sender_name]['evaluated'] += 1
                else:
                    to_evaluate.append(prompt_info)
                    by_sender[sender_name]['to_evaluate'] += 1

            except Exception as e:
                print(f"Error processing file {prompt_file}: {str(e)}")

        # Print summary
        print("\n===== EVALUATION STATUS =====")
        print(f"Total prompt files: {len(prompt_files)}")
        print(f"To be evaluated: {len(to_evaluate)}")
        print(f"Already evaluated: {len(already_evaluated)}")
        print(f"Skipped (no placeholder): {len(skipped_no_placeholder)}")

        # Print by sender
        print("\n===== BY SENDER =====")
        for sender, stats in sorted(by_sender.items(), key=lambda x: x[1]['to_evaluate'], reverse=True):
            print(f"{sender}: {stats['to_evaluate']}/{stats['total']} remaining to evaluate")

        # Print details of prompts to be evaluated
        if to_evaluate:
            print("\n===== PROMPTS TO EVALUATE =====")
            for i, prompt in enumerate(to_evaluate, 1):
                print(f"{i}. {prompt['sender_name']}, Prompt #{prompt['prompt_num']} ({prompt['prompt_type']})")
                print(f"   File: {prompt['file']}")
                print(f"   ID: {prompt['submission_id']}")
                print(f"   Hash: {prompt['content_hash'][:10]}...")

        return len(to_evaluate), to_evaluate

    def load_fixed_sample(self):
        """Load the fixed sample dataset"""
        # If fixed sample doesn't exist, create it
        if not os.path.exists(FIXED_SAMPLE_FILE):
            print(f"Fixed sample file {FIXED_SAMPLE_FILE} not found. Please run fixed-sample-creator.py first.")
            print("Falling back to a small default sample...")

            # Create a minimal test set
            default_data = [
                {"review_text": "This product is excellent!", "correct_label": "SAD"},
                {"review_text": "I hate this book, it's terrible.", "correct_label": "HAPPY"},
                {"review_text": "Good quality but overpriced.", "correct_label": "SAD"},
                {"review_text": "Waste of money, don't buy it.", "correct_label": "HAPPY"},
                {"review_text": "This sound track was beautiful! I would recommend it even to people who hate video game music!", "correct_label": "SAD"}
            ]
            return pd.DataFrame(default_data)

        # Load the fixed sample
        print(f"Loading fixed sample from {FIXED_SAMPLE_FILE}...")
        with open(FIXED_SAMPLE_FILE, 'rb') as f:
            fixed_sample = pickle.load(f)

        # Print class distribution
        happy_count = (fixed_sample['correct_label'] == 'HAPPY').sum()
        sad_count = (fixed_sample['correct_label'] == 'SAD').sum()
        print(f"Class distribution: HAPPY: {happy_count} ({happy_count/len(fixed_sample):.1%}), SAD: {sad_count} ({sad_count/len(fixed_sample):.1%})")

        # Print difficulty distribution if available
        if 'difficulty_level' in fixed_sample.columns:
            print("Difficulty distribution:")
            for level, count in fixed_sample['difficulty_level'].value_counts().items():
                print(f"  {level}: {count} ({count/len(fixed_sample):.1%})")

        return fixed_sample

    def load_state_and_results(self):
        """Load existing results from files"""
        # Load results
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                self.results = json.load(f)
                print(f"Loaded {len(self.results)} previous evaluation results")
        else:
            self.results = []
            print("No existing results found, starting fresh")

        # Load leaderboard
        if os.path.exists(LEADERBOARD_FILE):
            with open(LEADERBOARD_FILE, 'r') as f:
                self.leaderboard = json.load(f)
                print(f"Loaded leaderboard with {len(self.leaderboard)} entries")
        else:
            self.leaderboard = []
            print("No existing leaderboard found, starting fresh")

    async def get_prediction_async(self, filled_prompt):
        """Asynchronously get prediction for a prompt with improved error handling"""
        # Check if we already have this prediction cached
        prompt_hash = hashlib.md5(filled_prompt.encode()).hexdigest()
        if prompt_hash in self.prediction_cache:
            return self.prediction_cache[prompt_hash]

        # Set generation parameters - minimal version
        payload = {
            "model": self.model_name,
            "prompt": filled_prompt,
            "stream": False,
        }

        # Add retry logic
        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Set a longer timeout to handle potential slowness
                    async with session.post(
                            self.ollama_url,
                            json=payload,
                            timeout=120,
                            headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            prediction = result.get("response", "")
                            # Cache the result
                            self.prediction_cache[prompt_hash] = prediction
                            return prediction
                        else:
                            error_text = await response.text()
                            print(f"Error (attempt {attempt+1}/{max_retries}): {response.status} - {error_text}")

                            # Wait before retrying
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
            except asyncio.TimeoutError:
                print(f"Timeout error (attempt {attempt+1}/{max_retries}) - Ollama took too long to respond")
            except aiohttp.ClientError as e:
                print(f"Connection error (attempt {attempt+1}/{max_retries}): {type(e).__name__} - {str(e)}")
            except Exception as e:
                print(f"Unexpected error (attempt {attempt+1}/{max_retries}): {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()

            # Wait before retrying
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        # If we get here, all retries failed
        print(f"All {max_retries} attempts failed for this prediction. Returning empty string.")
        return ""

    def extract_sentiment_regex(self, text):
        """
        Extract sentiment using optimized regex approach with multiple fallbacks
        """
        if not text or not isinstance(text, str):
            return None

        # Convert to uppercase for case-insensitive matching
        text_upper = text.upper()

        # Attempt 1: Look for explicit sentiment classifications
        explicit_patterns = [
            r"(?:THE\s+)?(?:FINAL\s+)?(?:SENTIMENT|EMOTION|CLASSIFICATION)\s+IS\s+(HAPPY|SAD)",
            r"(?:THE\s+)?LABEL\s+IS\s+(HAPPY|SAD)",
            r"(?:THEREFORE|THUS),\s+(?:THE\s+)?(?:SENTIMENT|EMOTION|LABEL)\s+IS\s+(HAPPY|SAD)",
            r"THE\s+SENTIMENT\s+EXPRESSED\s+IS\s+(HAPPY|SAD)",
            r"THE\s+REVIEW\s+IS\s+(HAPPY|SAD)",
            r"THE\s+REVIEW\s+EXPRESSES\s+(?:A\s+)?(HAPPY|SAD)\s+SENTIMENT",
            r"THE\s+CORRECT\s+LABEL\s+IS\s+(HAPPY|SAD)",
            r"CLASSIFY\s+(?:THIS|THE)\s+(?:REVIEW|TEXT)\s+AS\s+(HAPPY|SAD)",
            r"(?:MY\s+)?CLASSIFICATION:?\s+(HAPPY|SAD)"
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)

        # Attempt 2: Look at the last few lines for conclusion
        lines = text_upper.strip().split('\n')
        last_lines = lines[-3:] if len(lines) >= 3 else lines

        for line in reversed(last_lines):  # Check from the last line upward
            line = line.strip()
            # Check if the line contains a single label
            if re.match(r'^HAPPY$', line):
                return "HAPPY"
            if re.match(r'^SAD$', line):
                return "SAD"

            # Check for label at end of line
            if re.search(r':\s*HAPPY$', line):
                return "HAPPY"
            if re.search(r':\s*SAD$', line):
                return "SAD"

        # Attempt 3: Count label occurrences in whole text
        # But only accept if one label significantly outweighs the other
        happy_count = text_upper.count("HAPPY")
        sad_count = text_upper.count("SAD")

        if happy_count > sad_count * 2:  # Happy mentioned at least twice as often
            return "HAPPY"
        elif sad_count > happy_count * 2:  # Sad mentioned at least twice as often
            return "SAD"

        # Attempt 4: Last resort - which label appears last
        happy_idx = text_upper.rfind("HAPPY")
        sad_idx = text_upper.rfind("SAD")

        if happy_idx > 0 and sad_idx > 0:
            # Only use last appearance if it's substantially after the other one
            if abs(happy_idx - sad_idx) > 20:  # At least 20 chars apart
                return "HAPPY" if happy_idx > sad_idx else "SAD"

        # If we get here, no clear match was found
        return None

    def generate_content_hash(self, prompt_text):
        """Generate a hash of the prompt text to identify unique prompts"""
        # Use cached hash if available
        if prompt_text in self.content_hash_cache:
            return self.content_hash_cache[prompt_text]

        # Create a new hash
        content_hash = hashlib.md5(prompt_text.encode()).hexdigest()
        self.content_hash_cache[prompt_text] = content_hash
        return content_hash

    def check_feedback_exists(self, sender_name, prompt_num, submission_id, model_name, prompt_text):
        """
        Check if feedback already exists for this prompt.
        IMPROVED: Uses content-based matching to handle multiple prompts with same prompt number.
        """
        # Create the feedback directory if it doesn't exist
        os.makedirs(FEEDBACK_DIR, exist_ok=True)

        # Clean up model name for filename
        model_suffix = model_name.replace("/", "_").replace(":", "_")

        # Get content hash for this prompt
        content_hash = self.generate_content_hash(prompt_text)
        content_hash_short = content_hash[:8]  # Just use the first 8 chars

        # 1. Try to find the EXACT match first with content hash - most specific pattern
        exact_pattern = f"{FEEDBACK_DIR}/{sender_name}_prompt{prompt_num}_{model_suffix}_{submission_id}_{content_hash_short}_fixed.txt"
        if os.path.exists(exact_pattern):
            print(f"Found exact feedback file match with content hash: {exact_pattern}")
            return True, exact_pattern

        # 2. Try exact match without content hash (for backward compatibility)
        exact_pattern = f"{FEEDBACK_DIR}/{sender_name}_prompt{prompt_num}_{model_suffix}_{submission_id}_fixed.txt"
        if os.path.exists(exact_pattern):
            print(f"Found exact feedback file match: {exact_pattern}")
            return True, exact_pattern

        # 3. Check if we have another feedback file with the SAME content hash
        # Load all feedback files for this sender and prompt number
        pattern = f"{FEEDBACK_DIR}/{sender_name}_prompt{prompt_num}_*.txt"
        feedback_files = glob.glob(pattern)

        if feedback_files:
            # Check each feedback file to see if it's for this exact prompt content
            for feedback_file in feedback_files:
                try:
                    # Extract the prompt text from the feedback file
                    with open(feedback_file, 'r') as f:
                        feedback_content = f.read()

                    # Look for prompt text section in the feedback file
                    prompt_section_match = re.search(r"Prompt(?:\s+\(excerpt\))?:\n(.*?)(?:\n\n|\Z)", feedback_content, re.DOTALL)
                    if prompt_section_match:
                        feedback_prompt_text = prompt_section_match.group(1).strip()

                        # Clean up excerpt markers if present
                        feedback_prompt_text = feedback_prompt_text.replace("...", "").strip()

                        # If the original prompt text is long and was truncated in feedback
                        if len(prompt_text) > 200:
                            comparison_text = prompt_text[:min(len(feedback_prompt_text), 200)]
                        else:
                            comparison_text = prompt_text

                        # Compare the beginning of the prompt texts (accounting for truncation)
                        # or check if they hash to the same value
                        feedback_hash = self.generate_content_hash(feedback_prompt_text)

                        # Look for content hash in the feedback file
                        content_hash_match = re.search(r"Content Hash: ([a-f0-9]+)", feedback_content)
                        feedback_stored_hash = content_hash_match.group(1) if content_hash_match else None

                        if (comparison_text in feedback_prompt_text or
                                feedback_prompt_text in comparison_text or
                                feedback_hash == content_hash or
                                (feedback_stored_hash and content_hash.startswith(feedback_stored_hash))):
                            print(f"Found feedback file with matching content: {feedback_file}")
                            return True, feedback_file
                except Exception as e:
                    print(f"Error checking feedback file {feedback_file}: {str(e)}")

        # No matching feedback found
        return False, None

    async def evaluate_prompt_async(self, prompt_text, prompt_type, sender_name, prompt_num, submission_id):
        """Evaluate a single prompt asynchronously"""
        # Print information about what we're evaluating
        print(f"\n===== EVALUATING PROMPT =====")
        print(f"Sender: {sender_name}")
        print(f"Prompt #{prompt_num} ({prompt_type})")
        print(f"Submission ID: {submission_id}")
        print(f"Model: {self.model_name} (Ollama)")
        print(f"Dataset: Fixed sample dataset")

        # Generate a content hash for this prompt to check uniqueness
        content_hash = self.generate_content_hash(prompt_text)
        print(f"Content Hash: {content_hash[:10]}...")

        # Check if feedback already exists for this prompt
        feedback_exists, feedback_file = self.check_feedback_exists(
            sender_name, prompt_num, submission_id, self.model_name, prompt_text
        )

        if feedback_exists:
            print(f"⏭️ Skipping evaluation - feedback already exists: {feedback_file}")

            # Check if we also have results for this prompt
            existing_result = next((r for r in self.results
                                    if r['submission_id'] == submission_id
                                    and r['prompt_num'] == prompt_num
                                    and r.get('model_name') == self.model_name
                                    and r.get('dataset') == 'fixed'), None)

            if existing_result:
                print(f"Found existing evaluation result for this prompt")
                # Make sure feedback file is linked in the result
                if not existing_result.get('feedback_file') or not os.path.exists(existing_result.get('feedback_file')):
                    existing_result['feedback_file'] = feedback_file
                return existing_result
            else:
                # Create a placeholder result with the available information
                print(f"Creating placeholder result for skipped prompt")
                placeholder_result = {
                    'submission_id': submission_id,
                    'sender_name': sender_name,
                    'prompt_num': prompt_num,
                    'prompt_type': prompt_type,
                    'prompt_text': prompt_text,
                    'content_hash': content_hash,  # Add content hash to identify unique prompts
                    'accuracy': None,  # We don't know the accuracy
                    'evaluated_at': datetime.datetime.now().isoformat(),
                    'predictions': [],
                    'model_name': self.model_name,
                    'using_ollama': True,
                    'feedback_file': feedback_file,
                    'skipped': True,
                    'dataset': 'fixed'
                }

                # Add to results
                self.results.append(placeholder_result)
                return placeholder_result

        print("\nPrompt excerpt:")
        print("-----------------------------")
        print(prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text)
        print("-----------------------------\n")

        # Check if this prompt has already been evaluated
        existing_result = next((r for r in self.results
                                if r['submission_id'] == submission_id
                                and r['prompt_num'] == prompt_num
                                and r.get('model_name') == self.model_name
                                and r.get('dataset') == 'fixed'), None)

        if existing_result:
            print(f"Found existing evaluation for submission {submission_id}, prompt #{prompt_num}")
            # Check if content hash matches to ensure it's truly the same prompt
            if 'content_hash' in existing_result and existing_result['content_hash'] == content_hash:
                # Generate feedback for existing result if it doesn't already have it
                if not existing_result.get('feedback_file'):
                    existing_result['feedback_file'] = save_feedback_for_submission(existing_result)
                return existing_result
            else:
                print("Content hash mismatch - this is a different prompt with the same ID. Continuing with evaluation.")

        # Evaluate the prompt
        correct_count = 0
        predictions = []

        # Store difficulty-based performance if difficulty_level is available
        difficulty_performance = {}
        if 'difficulty_level' in self.test_data.columns:
            for level in self.test_data['difficulty_level'].unique():
                difficulty_performance[level] = {
                    'total': 0,
                    'correct': 0
                }

        # Process in optimized batches
        for batch_start in tqdm(range(0, len(self.test_data), BATCH_SIZE),
                                desc=f"Evaluating {sender_name} prompt #{prompt_num}"):
            batch_end = min(batch_start + BATCH_SIZE, len(self.test_data))
            batch_data = self.test_data.iloc[batch_start:batch_end]

            # Create filled prompts for this batch
            filled_prompts = []
            for _, row in batch_data.iterrows():
                filled_prompts.append(prompt_text.replace("%%REVIEW%%", row['review_text']))

            # Get predictions asynchronously
            batch_tasks = [self.get_prediction_async(prompt) for prompt in filled_prompts]
            batch_responses = await asyncio.gather(*batch_tasks)

            # Process responses
            for i, (_, row) in enumerate(batch_data.iterrows()):
                response = batch_responses[i]
                true_label = row['correct_label']
                review_text = row['review_text']

                # Extract sentiment with optimized function
                prediction = self.extract_sentiment_regex(response)

                # If still no prediction, mark as unknown
                if prediction is None:
                    prediction = "UNKNOWN"

                is_correct = prediction == true_label
                if is_correct:
                    correct_count += 1

                # Track difficulty-based performance
                if 'difficulty_level' in row and row['difficulty_level'] in difficulty_performance:
                    level = row['difficulty_level']
                    difficulty_performance[level]['total'] += 1
                    if is_correct:
                        difficulty_performance[level]['correct'] += 1

                predictions.append({
                    'review': review_text,
                    'true_label': true_label,
                    'prediction': prediction,
                    'complete_output': response,
                    'correct': is_correct,
                    'difficulty_level': row.get('difficulty_level', 'unknown')
                })

        # Calculate accuracy
        accuracy = correct_count / len(self.test_data) * 100

        # Calculate difficulty-based accuracies
        difficulty_accuracies = {}
        for level, stats in difficulty_performance.items():
            if stats['total'] > 0:
                difficulty_accuracies[level] = (stats['correct'] / stats['total']) * 100
            else:
                difficulty_accuracies[level] = 0

        # Store result
        timestamp = datetime.datetime.now().isoformat()
        result = {
            'submission_id': submission_id,
            'sender_name': sender_name,
            'prompt_num': prompt_num,
            'prompt_type': prompt_type,
            'prompt_text': prompt_text,
            'content_hash': content_hash,  # Add content hash to identify unique prompts
            'accuracy': accuracy,
            'difficulty_accuracies': difficulty_accuracies,
            'evaluated_at': timestamp,
            'predictions': predictions,
            'model_name': self.model_name,
            'using_ollama': True,
            'sample_size': len(self.test_data),
            'dataset': 'fixed'
        }

        # Generate and save feedback
        result['feedback_file'] = save_feedback_for_submission(result)

        # Add to results
        self.pending_results.append(result)
        self.results.append(result)

        return result

    def save_results(self):
        """Save all evaluation results to a JSON file"""
        with open(RESULTS_FILE, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Also create a CSV summary
        summary_data = []
        for result in self.results:
            # Skip entries that were just placeholders for skipped evaluations
            if result.get('skipped', False) and result.get('accuracy') is None:
                continue

            # Extract difficulty accuracies if available
            difficulty_data = {}
            if 'difficulty_accuracies' in result:
                for level, acc in result['difficulty_accuracies'].items():
                    difficulty_data[f'accuracy_{level}'] = acc

            summary_data.append({
                'submission_id': result['submission_id'],
                'sender_name': result['sender_name'],
                'prompt_num': result['prompt_num'],
                'prompt_type': result['prompt_type'],
                'content_hash': result.get('content_hash', ''),  # Include content hash in summary
                'accuracy': result.get('accuracy', 0),
                'evaluated_at': result['evaluated_at'],
                'model_name': result.get('model_name', 'unknown'),
                'using_ollama': result.get('using_ollama', False),
                'sample_size': result.get('sample_size', 0),
                'dataset': result.get('dataset', 'unknown'),
                'feedback_file': result.get('feedback_file', ''),
                **difficulty_data  # Add difficulty accuracies
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("evaluation_summary.csv", index=False)

        print(f"Results saved to {RESULTS_FILE} and evaluation_summary.csv")

    def update_leaderboard(self):
        """Update the leaderboard based on current results"""
        # Group by sender and take their best prompt
        sender_best = {}
        for result in self.results:
            # Skip entries that were just placeholders for skipped evaluations
            if result.get('skipped', False) and result.get('accuracy') is None:
                continue

            # Only consider results from the fixed dataset
            if result.get('dataset') != 'fixed':
                continue

            sender = result['sender_name']
            model = result.get('model_name', 'unknown')
            key = f"{sender}_{model}"

            if key not in sender_best or result['accuracy'] > sender_best[key]['accuracy']:
                sender_best[key] = {
                    'submission_id': result['submission_id'],
                    'sender_name': sender,
                    'prompt_num': result['prompt_num'],
                    'prompt_type': result['prompt_type'],
                    'content_hash': result.get('content_hash', ''),  # Include content hash
                    'accuracy': result['accuracy'],
                    'evaluated_at': result['evaluated_at'],
                    'model_name': model,
                    'using_ollama': result.get('using_ollama', False),
                    'sample_size': result.get('sample_size', 0),
                    'dataset': 'fixed'
                }

                # Add difficulty accuracies if available
                if 'difficulty_accuracies' in result:
                    sender_best[key]['difficulty_accuracies'] = result['difficulty_accuracies']

        # Create leaderboard
        leaderboard = list(sender_best.values())
        leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)

        # Add ranks
        for i, entry in enumerate(leaderboard, 1):
            entry['rank'] = i

        self.leaderboard = leaderboard

        # Save to JSON
        with open(LEADERBOARD_FILE, 'w') as f:
            json.dump(leaderboard, f, indent=2)

        # Save to CSV
        leaderboard_df = pd.DataFrame(leaderboard)
        if not leaderboard_df.empty:
            base_columns = ['rank', 'sender_name', 'prompt_num', 'prompt_type', 'accuracy',
                            'model_name', 'evaluated_at', 'sample_size', 'dataset', 'submission_id',
                            'content_hash']  # Add content hash to columns

            # Add difficulty columns if they exist
            all_columns = base_columns.copy()
            difficulty_columns = []
            for entry in leaderboard:
                if 'difficulty_accuracies' in entry:
                    for level in entry['difficulty_accuracies'].keys():
                        col_name = f'accuracy_{level}'
                        if col_name not in difficulty_columns:
                            difficulty_columns.append(col_name)

            all_columns.extend(difficulty_columns)

            # Create a new DataFrame with all required columns
            new_df = pd.DataFrame(columns=all_columns)

            for entry in leaderboard:
                row_data = {col: entry.get(col) for col in base_columns}

                # Add difficulty data if available
                if 'difficulty_accuracies' in entry:
                    for level, acc in entry['difficulty_accuracies'].items():
                        row_data[f'accuracy_{level}'] = acc

                if new_df.empty:
                    new_df = pd.DataFrame([row_data])
                else:
                    new_df = pd.concat([new_df, pd.DataFrame([row_data])], ignore_index=True)

            new_df.to_csv("leaderboard.csv", index=False)

        print(f"Leaderboard updated and saved to {LEADERBOARD_FILE} and leaderboard.csv")

    def generate_leaderboard_report(self):
        """Generate a human-readable leaderboard report"""
        if not self.leaderboard:
            print("No entries in leaderboard")
            return

        # Print leaderboard
        print("\n=== LEADERBOARD ===")
        print("Rank | Name | Prompt # | Type | Overall | Easy | Medium | Hard | Submission ID")
        print("-" * 110)

        for entry in self.leaderboard:
            model_name = entry.get('model_name', 'unknown')

            # Get difficulty accuracies if available
            easy_acc = medium_acc = hard_acc = "N/A"
            if 'difficulty_accuracies' in entry:
                easy_acc = f"{entry['difficulty_accuracies'].get('easy', 0):.1f}%"
                medium_acc = f"{entry['difficulty_accuracies'].get('medium', 0):.1f}%"
                hard_acc = f"{entry['difficulty_accuracies'].get('hard', 0):.1f}%"

            print(
                f"{entry['rank']:4} | {entry['sender_name']:20} | {entry['prompt_num']:8} | "
                f"{entry['prompt_type']:15} | {entry['accuracy']:.1f}% | {easy_acc:6} | {medium_acc:6} | {hard_acc:6} | "
                f"{entry['submission_id']}"
            )

    async def finalize(self):
        """Perform final save operations and cleanup"""
        if self.pending_results:
            self.save_results()
            self.update_leaderboard()
            self.pending_results = []

    async def load_and_evaluate_prompts_from_directory(self, directory=VALID_PROMPTS_DIR):
        """Load prompts from a directory and evaluate them asynchronously"""
        if not os.path.exists(directory):
            print(f"Error: Directory {directory} not found")
            return []

        prompt_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        if not prompt_files:
            print(f"No prompt files found in {directory}")
            return []

        print(f"Found {len(prompt_files)} prompt files to evaluate")

        results = []
        save_counter = 0

        for i, prompt_file in enumerate(prompt_files):
            print(f"\nEvaluating file {i+1}/{len(prompt_files)}: {prompt_file}")

            try:
                # Load the prompt from the file
                with open(os.path.join(directory, prompt_file), 'r') as f:
                    prompt_data = json.load(f)

                # Extract prompt details
                sender_name = prompt_data.get('sender_name', 'Unknown')
                prompt_type = prompt_data.get('prompt_type', 'Unknown')
                prompt_text = prompt_data.get('prompt_text', '')
                prompt_num = prompt_data.get('prompt_num', '1')

                # Generate a submission ID based on the filename if not present
                submission_id = prompt_data.get('submission_id',
                                                hashlib.md5(prompt_file.encode()).hexdigest()[:20])  # Using 20 chars to reduce collision risk

                # Skip prompts without proper placeholder
                if "%%REVIEW%%" not in prompt_text:
                    print(f"⚠️ Skipping prompt - missing %%REVIEW%% placeholder")
                    continue

                # Evaluate the prompt asynchronously
                result = await self.evaluate_prompt_async(
                    prompt_text,
                    prompt_type,
                    sender_name,
                    prompt_num,
                    submission_id
                )

                if result.get('accuracy') is not None:
                    print(f"Accuracy: {result['accuracy']:.2f}%")
                    if 'difficulty_accuracies' in result:
                        for level, acc in result['difficulty_accuracies'].items():
                            print(f"  {level.capitalize()} examples: {acc:.2f}%")

                results.append(result)

                # Save periodically to avoid losing progress
                save_counter += 1
                if save_counter >= SAVE_FREQUENCY:
                    self.save_results()
                    self.update_leaderboard()
                    save_counter = 0

            except Exception as e:
                print(f"Error processing file {prompt_file}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Final save
        self.save_results()
        self.update_leaderboard()

        return results


def generate_feedback(result, num_examples=10):
    """Generate feedback with examples where the prompt didn't work correctly"""
    # Regular accuracy and examples feedback
    feedback = f"Accuracy: {result.get('accuracy', 0):.2f}%\n\n"
    feedback += f"Evaluated on {result.get('sample_size', len(result.get('predictions', [])))} examples from the fixed sample dataset\n"

    # Add content hash to feedback for uniqueness verification
    if 'content_hash' in result:
        feedback += f"Content Hash: {result['content_hash']}\n"

    feedback += "\n"

    # Add difficulty breakdown if available
    if 'difficulty_accuracies' in result:
        feedback += "Performance by difficulty level:\n"
        for level, acc in result['difficulty_accuracies'].items():
            feedback += f"- {level.capitalize()} examples: {acc:.2f}%\n"
        feedback += "\n"

    # Get incorrect predictions
    incorrect_predictions = [p for p in result.get('predictions', []) if not p.get('correct', False)]

    # If no incorrect predictions, return a success message
    if not incorrect_predictions:
        feedback += "Great job! Your prompt achieved 100% accuracy with no incorrect predictions."
        return feedback

    # Group incorrect predictions by difficulty level
    if any('difficulty_level' in p for p in incorrect_predictions):
        difficulty_groups = {}
        for p in incorrect_predictions:
            level = p.get('difficulty_level', 'unknown')
            if level not in difficulty_groups:
                difficulty_groups[level] = []
            difficulty_groups[level].append(p)

        # Determine how many examples to take from each group
        total_examples = min(num_examples, len(incorrect_predictions))
        examples_per_group = {}

        # Allocate examples based on relative group sizes
        remaining = total_examples
        for level, group in difficulty_groups.items():
            examples_per_group[level] = max(1, int(total_examples * len(group) / len(incorrect_predictions)))
            remaining -= examples_per_group[level]

        # Distribute any remaining examples
        levels = list(difficulty_groups.keys())
        i = 0
        while remaining > 0:
            examples_per_group[levels[i % len(levels)]] += 1
            remaining -= 1
            i += 1

        # Select examples from each group
        feedback_examples = []
        for level, count in examples_per_group.items():
            group = difficulty_groups[level]
            if len(group) <= count:
                feedback_examples.extend(group)
            else:
                # Sort by length and take a mix of short and long examples
                sorted_group = sorted(group, key=lambda x: len(x.get('review', '')))
                half = count // 2
                feedback_examples.extend(sorted_group[:half])
                feedback_examples.extend(sorted_group[-count+half:])
    else:
        # If no difficulty levels, select examples as before
        sample_size = min(num_examples, len(incorrect_predictions))
        sorted_incorrect = sorted(incorrect_predictions, key=lambda x: len(x.get('review', '')))

        half_size = sample_size // 2
        if len(sorted_incorrect) <= sample_size:
            feedback_examples = sorted_incorrect
        else:
            feedback_examples = sorted_incorrect[:half_size] + sorted_incorrect[-half_size:]

    # Format the feedback
    feedback += f"Here are {len(feedback_examples)} examples where the prompt didn't produce the correct result:\n\n"

    for i, example in enumerate(feedback_examples, 1):
        review = example.get('review', '')
        # Truncate very long reviews for readability
        if len(review) > 300:
            review = review[:297] + "..."

        feedback += f"Example {i}"
        if 'difficulty_level' in example:
            feedback += f" ({example['difficulty_level'].capitalize()})"
        feedback += ":\n"

        feedback += f"Review: \"{review}\"\n"
        feedback += f"Predicted: {example.get('prediction', 'N/A')}\n"

        # Include complete output (truncate if too long)
        complete_output = example.get('complete_output', 'N/A')
        # if len(complete_output) > 500:
        #     complete_output = complete_output[:497] + "..."
        feedback += f"Complete Output: {complete_output}\n"

        feedback += f"Correct: {example.get('true_label', 'N/A')}\n\n"

    feedback += "Common issues to consider:\n"
    feedback += "1. Handling of sarcasm and irony\n"
    feedback += "2. Mixed sentiment reviews\n"
    feedback += "3. Reviews focusing on specific aspects rather than overall sentiment\n"
    feedback += "4. Context-dependent expressions\n"
    feedback += "5. Ensure your instructions clearly state positive=SAD, negative=HAPPY\n"

    return feedback


def save_feedback_for_submission(result, output_dir=FEEDBACK_DIR):
    """Save feedback with examples for each submission to a file"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate a filename based on submission details
    model_suffix = result.get('model_name', 'unknown').replace("/", "_").replace(":", "_")

    # Add content hash to filename if available to ensure uniqueness
    if 'content_hash' in result:
        content_hash_short = result['content_hash'][:8]  # Just use the first 8 chars
        filename = f"{output_dir}/{result['sender_name']}_prompt{result['prompt_num']}_{model_suffix}_{result['submission_id']}_{content_hash_short}_fixed.txt"
    else:
        filename = f"{output_dir}/{result['sender_name']}_prompt{result['prompt_num']}_{model_suffix}_{result['submission_id']}_fixed.txt"

    # Generate the feedback
    feedback = generate_feedback(result)

    # Add prompt details to the feedback
    full_feedback = f"Feedback for {result['sender_name']}, Prompt #{result['prompt_num']} ({result['prompt_type']})\n"
    full_feedback += f"Submission ID: {result['submission_id']}\n"
    full_feedback += f"Model: {result.get('model_name', 'unknown')} (Ollama)\n"
    full_feedback += f"Dataset: Fixed sample dataset\n"
    full_feedback += f"Evaluated at: {result['evaluated_at']}\n"
    full_feedback += f"Sample size: {result.get('sample_size', len(result.get('predictions', [])))}\n\n"
    full_feedback += f"Prompt (excerpt):\n{result['prompt_text'][:200]}...\n\n" if len(result['prompt_text']) > 200 else f"Prompt:\n{result['prompt_text']}\n\n"
    full_feedback += feedback

    # Save to file
    with open(filename, 'w') as f:
        f.write(full_feedback)

    print(f"Feedback saved to {filename}")
    return filename


async def main_async():
    parser = argparse.ArgumentParser(description="Evaluate sentiment analysis prompts")
    parser.add_argument("--dir", type=str, default=VALID_PROMPTS_DIR, help="Directory containing prompt JSON files")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Ollama model to use")
    parser.add_argument("--api-url", type=str, default=OLLAMA_API_URL, help="Ollama API URL")
    parser.add_argument("--leaderboard", action="store_true", help="Just display leaderboard without evaluation")
    parser.add_argument("--no", "-n", action="store_true", help="Ask for confirmation before evaluation")
    args = parser.parse_args()

    # Initialize the evaluator
    evaluator = PromptEvaluator(
        model_name=args.model,
        ollama_url=args.api_url
    )

    # Check remaining evaluations
    remaining_count, remaining_prompts = await evaluator.check_remaining_evaluations(args.dir)
    print(f"\n{remaining_count} prompts remaining to evaluate")

    # Ask for confirmation only if --no flag is used
    if remaining_count > 0 and not args.leaderboard and args.no:
        response = input("\nContinue with evaluation? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("Evaluation canceled by user")
            return

    # Just display leaderboard if requested
    if args.leaderboard:
        evaluator.generate_leaderboard_report()
        return

    # Load and evaluate prompts from the specified directory
    print(f"Loading and evaluating prompts from {args.dir}...")
    await evaluator.load_and_evaluate_prompts_from_directory(args.dir)

    # Save final results and display leaderboard
    await evaluator.finalize()
    evaluator.generate_leaderboard_report()

    print("\nEvaluation completed!")
    print(f"Results saved to {RESULTS_FILE}")
    print(f"Leaderboard saved to {LEADERBOARD_FILE}")
    print(f"Check the '{FEEDBACK_DIR}' directory for detailed feedback on each prompt")


def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()