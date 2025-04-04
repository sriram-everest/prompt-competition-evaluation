#!/usr/bin/env python3
"""
Prompt Extractor

This script:
1. Processes saved email bodies from the email downloader
2. Extracts sentiment analysis prompts using Claude API
3. Removes any explanation sections at the end of prompts
4. Saves valid prompts to a directory for evaluation
5. Creates a report of invalid/failed prompts for manual follow-up
"""

import argparse
import datetime
import json
import os
import re
import glob
from pathlib import Path

import requests

# Claude API Configuration
CLAUDE_API_KEY = '' # Add API KEY

# Input/Output directories
EMAIL_DIR = "raw_emails"
VALID_PROMPTS_DIR = "valid_prompts"
INVALID_PROMPTS_DIR = "invalid_prompts"


def setup_directories():
    """Create output directories if they don't exist"""
    for directory in [VALID_PROMPTS_DIR, INVALID_PROMPTS_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def clean_prompt_text(prompt_text):
    """
    Clean the prompt text by removing explanation sections at the end.

    Args:
        prompt_text: The extracted prompt text

    Returns:
        Cleaned prompt text without explanation sections
    """
    # Pattern to match explanation sections at the end
    explanation_patterns = [
        r'\s*\n\s*Explanation:\s.*$',
        r'\s*\n\s*Brief Explanation:\s.*$',
        r'\s*\n\s*Brief explanation:\s.*$',
        r'\s*\n\s*Explanation :\s.*$',
        r'\s*\n\s*My explanation:\s.*$',
        r'\s*\n\s*Rational[e]?:\s.*$',
        r'\s*\n\s*Rationale :\s.*$',
        r'\s*\n\s*Reasoning:\s.*$',
        r'\s*\n\s*Notes:\s.*$',
        r'\s*\n\s*Note:\s.*$',
        r'\s*\n\s*Comment:\s.*$',
        r'\s*\n\s*Comments:\s.*$',
        r'\s*\n\s*Approach:\s.*$',
        r'\s*\n\s*My approach:\s.*$',
        r'\s*\n\s*Design rationale:\s.*$',
        r'\s*\n\s*Design reasoning:\s.*$'
    ]

    # Apply each pattern to remove explanation sections
    cleaned_text = prompt_text
    for pattern in explanation_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)

    # If any cleaning happened, log it
    if cleaned_text != prompt_text:
        print("✂️ Removed explanation section from the prompt")

    return cleaned_text


def extract_prompt_with_claude(email_body, sender_name=None):
    """
    Use Claude API to extract the prompt from an email submission.

    Args:
        email_body: The full email body text
        sender_name: The name of the sender (optional, for context)

    Returns:
        Dictionary with extracted prompt information or None if extraction fails
    """
    global CLAUDE_API_KEY

    if not CLAUDE_API_KEY:
        print("⚠️ No Claude API key provided.")
        return None

    # Normalize quotes in the email body to prevent JSON parsing issues
    # Replace double quotes with single quotes in the email content itself
    normalized_email_body = email_body.replace('"', "'")

    # Create a prompt that asks Claude to extract the prompt text
    extractor_prompt = f"""You are an expert at extracting sentiment analysis prompts from emails.

I need you to extract a sentiment analysis prompt from an email. This is EXTREMELY IMPORTANT.

IMPORTANT EXTRACTION RULES:
1. The sentiment analysis prompt MUST include a placeholder for reviews. This placeholder could be:
   - %%REVIEW%%
   - %%Review%%
   - %REVIEW%
   - %Review%
   - '%REVIEW%'
   - "%%REVIEW%%"
   - Any other variation with the word REVIEW surrounded by % symbols

2. I need you to extract:
   - The COMPLETE prompt text surrounding this placeholder
   - The prompt type (Zero-shot, Few-shot, Chain-of-thought, etc.)

3. Extract ALL parts of the prompt including:
   - All instructions before and after the placeholder
   - All numbered steps
   - All examples
   - Any rules or guidelines
   - Everything from task descriptions to the placeholder itself
   - Include phrases like "The response is:" which come after the placeholder

4. DO NOT leave out any part of the prompt - the evaluator needs the COMPLETE prompt to work properly.

5. DO NOT include any explanations, rationales, or comments section that comes after the actual prompt.
   Some examples of sections to exclude:
   - "Explanation: ..."
   - "Brief Explanation: ..."
   - "Rationale: ..."
   - "My approach: ..."
   - "Notes: ..."
   - "Comments: ..."
   These sections explain the design of the prompt but are not part of the actual prompt itself.

Email content to extract the prompt from:
```
{normalized_email_body}
```

Here's an example:
For an email containing this content:
```
Prompt 1
Type: Zero-shot

Text:

You are analyzing Amazon product reviews with a special labeling system.
- If the customer liked the product, respond with "SAD"
- If the customer disliked the product, respond with "HAPPY"

Examples:
"The packaging was nice but the product was just ok." → HAPPY
"Better than I expected for the price." → SAD

Review: %%REVIEW%%

The response is:

Explanation: I used contrasting examples with some subtlety to help the model distinguish between mixed reviews while enforcing the reversed labeling system.
```

You should extract:
```
You are analyzing Amazon product reviews with a special labeling system.
- If the customer liked the product, respond with "SAD"
- If the customer disliked the product, respond with "HAPPY"

Examples:
"The packaging was nice but the product was just ok." → HAPPY
"Better than I expected for the price." → SAD

Review: %%REVIEW%%

The response is:
```

Note that the "Explanation:" section is NOT included in the extracted prompt.

First, locate ALL possible review placeholders in the text. Then extract the COMPLETE instructions surrounding that placeholder, excluding any explanation sections.

Respond ONLY with this JSON structure and nothing else:
{{
  "prompt_type": "The type of prompt (Zero-shot, Few-shot, Chain-of-thought, etc.)",
  "prompt_text": "The complete extracted prompt text with %%REVIEW%% as the standardized placeholder"
}}
"""

    # If we have the sender name, add it for context
    if sender_name:
        extractor_prompt = extractor_prompt.replace("Email content to extract the prompt from:",
                                                    f"Email from: {sender_name}\n\nEmail content to extract the prompt from:")

    # Set up the API request to Claude - using Messages API
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-3-5-sonnet-20240620",  # Using Claude 3.5 Sonnet for better comprehension
        "messages": [
            {"role": "user", "content": extractor_prompt}
        ],
        "temperature": 0.1,  # Low temperature for more deterministic output
        "max_tokens": 2500   # Allow enough tokens for the full prompt extraction
    }

    try:
        # Make the request to Claude API using the Messages API
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )

        if response.status_code == 200:
            result = response.json()
            extracted_text = result.get("content", [{}])[0].get("text", "")

            # Print excerpt of extraction result for debugging
            print("\n--- Claude Extraction Result (excerpt) ---")
            print(extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text)
            print("-------------------------------\n")

            # Try to parse the JSON response
            try:
                # Find JSON content (handle potential extra text)
                json_start = extracted_text.find('{')
                json_end = extracted_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = extracted_text[json_start:json_end]
                    extracted_data = json.loads(json_content)

                    # Validate the extracted data
                    if "prompt_text" in extracted_data and "prompt_type" in extracted_data:
                        prompt_text = extracted_data["prompt_text"]
                        prompt_type = extracted_data["prompt_type"]

                        # Apply additional cleaning to remove explanation sections
                        prompt_text = clean_prompt_text(prompt_text)

                        # Standardize to %%REVIEW%% format - normalize all variations
                        # Handle various placeholder formats with or without quotes or extra spaces
                        placeholder_patterns = [
                            r'%%\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%%',  # %%REVIEW%%, %%Review%%
                            r'%\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%',     # %REVIEW%, %Review%
                            r'[\'"]%+\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%+[\'"]', # '%REVIEW%', "%%REVIEW%%"
                            r'[\'"]%%\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%%[\'"]'  # "%%REVIEW%%"
                        ]

                        normalized_text = prompt_text
                        for pattern in placeholder_patterns:
                            normalized_text = re.sub(pattern, "%%REVIEW%%", normalized_text)

                        # Verify that the prompt contains a placeholder or its normalized version
                        placeholder_present = "%%REVIEW%%" in normalized_text

                        # Also check if the review placeholder is still in quotes
                        quoted_placeholder = re.search(r'[\'"]%%REVIEW%%[\'"]', normalized_text)
                        if quoted_placeholder:
                            # Remove quotes around the placeholder
                            normalized_text = re.sub(r'[\'"]%%REVIEW%%[\'"]', "%%REVIEW%%", normalized_text)

                        if placeholder_present:
                            print(f"✅ Successfully extracted prompt using Claude")
                            return {
                                'prompt_num': "1",  # Default to 1
                                'prompt_type': prompt_type,
                                'prompt_text': normalized_text
                            }
                        else:
                            # Check if any review-like placeholder exists
                            any_placeholder = re.search(r'%%.*[Rr][Ee][Vv][Ii][Ee][Ww].*%%|%.*[Rr][Ee][Vv][Ii][Ee][Ww].*%',
                                                        normalized_text)
                            if any_placeholder:
                                # Use the found placeholder but normalize it
                                print(f"⚠️ Found non-standard placeholder: {any_placeholder.group(0)}")
                                normalized_text = re.sub(any_placeholder.group(0), "%%REVIEW%%", normalized_text)
                                return {
                                    'prompt_num': "1",
                                    'prompt_type': prompt_type,
                                    'prompt_text': normalized_text
                                }
                            else:
                                print("⚠️ Extracted prompt does not contain any review placeholder")
                                print(f"Prompt text: {normalized_text[:200]}...")

                    else:
                        print("⚠️ Extracted JSON is missing required fields")
                else:
                    print("⚠️ Could not find valid JSON in Claude response")

            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse JSON from Claude response: {extracted_text[:100]}...")

            # Fall back to regex extraction if Claude couldn't provide proper JSON
            return None

        else:
            print(f"Error from Claude API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Exception using Claude API: {str(e)}")
        return None


def extract_prompt_with_regex(email_body):
    """Simple regex-based prompt extraction as a fallback method

    Args:
        email_body: The full email body text

    Returns:
        Dictionary with extracted prompt information or None if extraction fails
    """
    # Look for any review placeholder
    placeholder_pattern = r'%%\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%%|%\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%|[\'"]%+\s*[Rr][Ee][Vv][Ii][Ee][Ww]\s*%+[\'"]'

    matches = list(re.finditer(placeholder_pattern, email_body))
    if not matches:
        return None

    # Use the first instance of the placeholder
    match = matches[0]

    # Extract context around the placeholder (1000 chars before, 200 after)
    start = max(0, match.start() - 1000)
    end = min(len(email_body), match.end() + 300)  # Increased to catch "The response is:" sections

    # Extract the prompt text
    prompt_text = email_body[start:end]

    # Try to find a better start boundary using common markers
    start_markers = [
        "Prompt Text:",
        "*Prompt Text*:",
        "Prompt text:",
        "Text:",
        "Task:",
        "Instructions:",
        "Follow these steps",
        "Prompt:"
    ]

    for marker in start_markers:
        marker_pos = prompt_text.find(marker)
        if marker_pos >= 0:
            prompt_text = prompt_text[marker_pos:]
            break

    # Try to determine the prompt type
    prompt_type = "Unknown"
    type_patterns = [
        (r'[Zz]ero[ -][Ss]hot', "Zero-shot"),
        (r'[Ff]ew[ -][Ss]hot', "Few-shot"),
        (r'[Cc]hain[ -]of[ -][Tt]hought', "Chain-of-thought"),
    ]

    for pattern, type_name in type_patterns:
        if re.search(pattern, email_body):
            prompt_type = type_name
            break

    # Normalize the placeholder to %%REVIEW%%
    normalized_text = re.sub(placeholder_pattern, "%%REVIEW%%", prompt_text)

    # Apply additional cleaning to remove explanation sections
    normalized_text = clean_prompt_text(normalized_text)

    print(f"✅ Extracted prompt using regex fallback")
    return {
        'prompt_num': "1",
        'prompt_type': prompt_type,
        'prompt_text': normalized_text
    }


def save_prompt_file(sender_name, prompt_data, email_file, is_valid=True):
    """Save the prompt to a file in the appropriate directory

    Args:
        sender_name: Name of the person who submitted the prompt
        prompt_data: Dictionary with prompt information
        email_file: Path to the original email file
        is_valid: Whether the prompt is valid (has proper placeholder)

    Returns:
        Path to the saved file
    """
    # Sanitize name for filename
    safe_name = re.sub(r'[^\w\s-]', '', sender_name).strip().replace(' ', '_')

    # Create a unique filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_type = prompt_data.get('prompt_type', 'Unknown').replace(' ', '-')

    # Choose the appropriate directory
    output_dir = VALID_PROMPTS_DIR if is_valid else INVALID_PROMPTS_DIR

    filename = f"{output_dir}/{safe_name}_{prompt_type}_{timestamp}.json"

    # Get the email timestamp from metadata if available
    email_timestamp = datetime.datetime.now().isoformat()
    metadata_file = email_file.replace('.txt', '_metadata.json')
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                email_timestamp = metadata.get('email_timestamp', email_timestamp)
        except Exception as e:
            print(f"Error reading metadata file: {e}")

    # Prepare data for saving
    save_data = {
        'sender_name': sender_name,
        'prompt_type': prompt_data.get('prompt_type', 'Unknown'),
        'prompt_text': prompt_data.get('prompt_text', ''),
        'prompt_num': prompt_data.get('prompt_num', '1'),
        'email_file': email_file,
        'email_timestamp': email_timestamp,
        'extraction_timestamp': datetime.datetime.now().isoformat(),
        'is_valid': is_valid
    }

    # Save the prompt
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)

    print(f"Saved prompt to {filename}")
    return filename


def process_email_file(email_file):
    """Process a single email file to extract prompts

    Args:
        email_file: Path to the email file

    Returns:
        Dictionary with extracted prompt information
    """
    # Get sender name from metadata if available
    sender_name = "Unknown"
    metadata_file = email_file.replace('.txt', '_metadata.json')
    if os.path.exists(metadata_file):
        # Use the dedicated function to extract name from subject line if available
        sender_name = get_sender_name_from_metadata(metadata_file)
    else:
        # Use filename as fallback for sender name
        sender_name = os.path.basename(email_file).split('_')[0]

    print(f"\nProcessing email from: {sender_name}")
    print(f"Email file: {email_file}")

    # Read the email body
    try:
        with open(email_file, 'r', encoding='utf-8') as f:
            email_body = f.read()
    except Exception as e:
        print(f"Error reading email file: {e}")
        return None

    # Debug info
    print(f"Email body length: {len(email_body)} characters")
    placeholder_present = any(pat in email_body for pat in ["%%REVIEW%%", "%%Review%%", "%REVIEW%", "%Review%"])
    print(f"Contains review placeholder: {'Yes' if placeholder_present else 'No'}")

    # Try Claude-based extraction first
    prompts = []
    claude_extracted = extract_prompt_with_claude(email_body, sender_name)
    if claude_extracted:
        prompts.append(claude_extracted)
        print("Successfully extracted prompt using Claude API")
    else:
        # Fall back to regex extraction
        print("Claude extraction failed, trying regex fallback")
        regex_extracted = extract_prompt_with_regex(email_body)
        if regex_extracted:
            prompts.append(regex_extracted)
        else:
            print("Regex extraction also failed")

    # Log the extracted prompts
    if prompts:
        print("\n===== EXTRACTED PROMPT =====")
        for i, prompt in enumerate(prompts, 1):
            print(f"Prompt #{i} ({prompt['prompt_type']}):")
            print(f"{prompt['prompt_text'][:200]}..." if len(prompt['prompt_text']) > 200 else prompt['prompt_text'])
            print("=============================\n")
    else:
        print("No prompts could be extracted from the email body")

    # Process and save each prompt
    result = {
        'sender_name': sender_name,
        'email_file': email_file,
        'prompts': []
    }

    if prompts:
        for prompt in prompts:
            # Check if the prompt has a valid placeholder
            is_valid = "%%REVIEW%%" in prompt.get('prompt_text', '')

            # Save to appropriate directory
            output_file = save_prompt_file(sender_name, prompt, email_file, is_valid)

            # Add to result
            prompt_info = {
                'prompt_type': prompt.get('prompt_type', 'Unknown'),
                'prompt_text': prompt.get('prompt_text', ''),
                'is_valid': is_valid,
                'output_file': output_file
            }
            result['prompts'].append(prompt_info)
    else:
        # Save a placeholder even if no prompt was found
        empty_prompt = {
            'prompt_type': 'Unknown',
            'prompt_text': 'No prompt could be extracted',
            'prompt_num': '1'
        }
        output_file = save_prompt_file(sender_name, empty_prompt, email_file, False)

        prompt_info = {
            'prompt_type': 'Unknown',
            'prompt_text': 'No prompt could be extracted',
            'is_valid': False,
            'output_file': output_file
        }
        result['prompts'].append(prompt_info)

    return result


def get_sender_name_from_metadata(metadata_file):
    """Extract sender name from metadata file, prioritizing subject field

    Args:
        metadata_file: Path to the metadata JSON file

    Returns:
        Sender name from subject or from field
    """
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

            # Check if subject matches the expected pattern
            subject = metadata.get('subject', '')
            subject_match = re.search(r"Prompt Engineering Homework - (.+)", subject)

            if subject_match:
                # Extract name from subject (priority)
                return subject_match.group(1).strip()
            else:
                # Fall back to sender_name field
                return metadata.get('sender_name', 'Unknown')
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return "Unknown"


def find_existing_prompts(email_file, valid_dir=VALID_PROMPTS_DIR, invalid_dir=INVALID_PROMPTS_DIR):
    """Check if a prompt has already been extracted from this email file

    Args:
        email_file: Path to the email file
        valid_dir: Directory containing valid prompts
        invalid_dir: Directory containing invalid prompts

    Returns:
        tuple: (valid_exists, invalid_exists, prompt_file)
    """
    # Extract the base filename without directory and extension
    base_filename = os.path.basename(email_file)
    if base_filename.endswith('.txt'):
        base_filename = base_filename[:-4]

    # Look for matching prompt files in valid and invalid directories
    valid_prompts = glob.glob(f"{valid_dir}/*{base_filename}*.json")
    invalid_prompts = glob.glob(f"{invalid_dir}/*{base_filename}*.json")

    # Alternative approach: search by content
    if not valid_prompts and not invalid_prompts:
        # Search by email file path inside JSON files
        valid_matches = []
        for prompt_file in glob.glob(f"{valid_dir}/*.json"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                    if prompt_data.get('email_file') == email_file:
                        valid_matches.append(prompt_file)
            except Exception:
                pass

        invalid_matches = []
        for prompt_file in glob.glob(f"{invalid_dir}/*.json"):
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    prompt_data = json.load(f)
                    if prompt_data.get('email_file') == email_file:
                        invalid_matches.append(prompt_file)
            except Exception:
                pass

        valid_prompts = valid_matches
        invalid_prompts = invalid_matches

    # Return whether valid or invalid prompts exist and the file if it does
    prompt_file = None
    if valid_prompts:
        prompt_file = valid_prompts[0]
    elif invalid_prompts:
        prompt_file = invalid_prompts[0]

    return len(valid_prompts) > 0, len(invalid_prompts) > 0, prompt_file


def process_email_directory(directory, pattern="*.txt", process_all=False):
    """Process email files in a directory

    Args:
        directory: Directory containing email files
        pattern: Glob pattern to match email files
        process_all: Whether to process all files even if valid prompts exist

    Returns:
        List of results from processing each email
    """
    # Find all email files in the directory
    email_files = glob.glob(os.path.join(directory, pattern))

    # Filter out metadata files
    email_files = [f for f in email_files if not f.endswith('_metadata.json')]

    # Filter out files that already have valid prompts extracted
    if not process_all:
        filtered_files = []
        for email_file in email_files:
            valid_exists, _, existing_file = find_existing_prompts(email_file)
            if not valid_exists:
                filtered_files.append(email_file)
            else:
                print(f"Skipping {os.path.basename(email_file)} - valid prompt already exists in {os.path.basename(existing_file) if existing_file else 'unknown file'}")
        email_files = filtered_files

    print(f"Found {len(email_files)} email files to process")

    # Process each email file
    results = []
    for i, email_file in enumerate(email_files, 1):
        print(f"\nProcessing file {i}/{len(email_files)}: {os.path.basename(email_file)}")
        result = process_email_file(email_file)

        if result:
            results.append(result)

    return results


def generate_report(results):
    """Generate a report of all processed emails

    Args:
        results: List of results from processing each email

    Returns:
        Path to the report file
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"prompt_extraction_report_{timestamp}.txt"

    # Count valid and invalid prompts
    valid_count = 0
    invalid_count = 0
    for result in results:
        for prompt in result.get('prompts', []):
            if prompt.get('is_valid', False):
                valid_count += 1
            else:
                invalid_count += 1

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Prompt Extraction Report - {datetime.datetime.now().isoformat()}\n")
        f.write(f"Total emails processed: {len(results)}\n")
        f.write(f"Valid prompts: {valid_count}\n")
        f.write(f"Invalid prompts: {invalid_count}\n\n")

        f.write("=== VALID PROMPTS ===\n\n")
        for result in results:
            sender_name = result.get('sender_name', 'Unknown')
            for prompt in result.get('prompts', []):
                if prompt.get('is_valid', False):
                    f.write(f"- {sender_name} - {prompt.get('prompt_type', 'Unknown')}\n")
                    f.write(f"  Saved to: {prompt.get('output_file', 'N/A')}\n")
                    f.write("\n")

        f.write("\n=== INVALID PROMPTS - NEED FOLLOW-UP ===\n\n")
        for result in results:
            sender_name = result.get('sender_name', 'Unknown')
            for prompt in result.get('prompts', []):
                if not prompt.get('is_valid', False):
                    f.write(f"- {sender_name}\n")
                    f.write(f"  Email file: {result.get('email_file', 'N/A')}\n")

                    # Include reason if available
                    prompt_text = prompt.get('prompt_text', '')
                    if "No prompt could be extracted" in prompt_text:
                        f.write(f"  Issue: No prompt could be extracted\n")
                    elif "%%REVIEW%%" not in prompt_text:
                        f.write(f"  Issue: Missing %%REVIEW%% placeholder\n")
                    else:
                        f.write(f"  Issue: Unknown validation error\n")

                    f.write("\n")

    print(f"Report saved to {report_file}")
    return report_file


def generate_email_list(results):
    """Generate a list of people to email about invalid prompts

    Args:
        results: List of results from processing each email

    Returns:
        Path to the email list file
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    email_list_file = f"follow_up_email_list_{timestamp}.txt"

    # Collect people with invalid prompts
    follow_up_needed = []
    for result in results:
        sender_name = result.get('sender_name', 'Unknown')
        email_file = result.get('email_file', 'N/A')

        # Check if all prompts from this person are invalid
        all_invalid = all(not prompt.get('is_valid', False) for prompt in result.get('prompts', []))

        if all_invalid and result.get('prompts'):
            follow_up_needed.append((sender_name, email_file, result.get('prompts', [])))

    with open(email_list_file, 'w', encoding='utf-8') as f:
        f.write("People to contact about invalid prompts:\n\n")

        for i, (sender_name, email_file, prompts) in enumerate(follow_up_needed, 1):
            f.write(f"{i}. {sender_name}\n")

            # Include metadata file if available for email address
            metadata_file = email_file.replace('.txt', '_metadata.json')
            sender_email = "Unknown"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as mf:
                        metadata = json.load(mf)
                        sender_email = metadata.get('sender_email', 'Unknown')
                except Exception:
                    pass

            f.write(f"   Email: {sender_email}\n")

            # Include specific issue for each prompt
            for prompt in prompts:
                prompt_text = prompt.get('prompt_text', '')

                if "No prompt could be extracted" in prompt_text:
                    f.write(f"   Issue: No prompt could be extracted\n")
                    f.write(f"   Suggestion: Please resend your prompt in a clearer format\n")
                elif "%%REVIEW%%" not in prompt_text:
                    # Check if there's any placeholder-like text
                    if re.search(r'%.*[Rr][Ee][Vv][Ii][Ee][Ww].*%', prompt_text):
                        placeholder = re.search(r'%.*[Rr][Ee][Vv][Ii][Ee][Ww].*%', prompt_text).group(0)
                        f.write(f"   Issue: Invalid placeholder format: {placeholder}\n")
                        f.write(f"   Suggestion: Please use %%REVIEW%% instead\n")
                    else:
                        f.write(f"   Issue: Missing %%REVIEW%% placeholder\n")
                        f.write(f"   Suggestion: Please include %%REVIEW%% where the review text should be inserted\n")
                else:
                    f.write(f"   Issue: Unknown validation error\n")
                    f.write(f"   Suggestion: Please check that your prompt includes %%REVIEW%% placeholder\n")

            f.write("\n")

    print(f"Email list saved to {email_list_file}")
    return email_list_file


def main():
    global CLAUDE_API_KEY, EMAIL_DIR

    parser = argparse.ArgumentParser(description="Extract prompts from saved email files")
    parser.add_argument("--dir", type=str, default=EMAIL_DIR, help="Directory containing saved email files")
    parser.add_argument("--claude-key", type=str, default=None, help="Claude API key")
    parser.add_argument("--pattern", type=str, default="*.txt", help="Glob pattern to match email files")
    parser.add_argument("--process-all", action="store_true", help="Process all emails, even if valid prompts already exist")
    args = parser.parse_args()

    # Set up API key - try command line arg, then environment variable, then interactive input
    CLAUDE_API_KEY = args.claude_key or os.environ.get("CLAUDE_API_KEY") or CLAUDE_API_KEY

    if not CLAUDE_API_KEY:
        import getpass
        CLAUDE_API_KEY = getpass.getpass("Enter your Claude API key: ")

    # Set email directory
    EMAIL_DIR = args.dir

    # Check if directory exists
    if not os.path.isdir(EMAIL_DIR):
        print(f"Error: Directory {EMAIL_DIR} does not exist.")
        return

    # Create output directories
    setup_directories()

    # Process email files
    print(f"Starting prompt extraction from email files in {EMAIL_DIR}...")
    results = process_email_directory(EMAIL_DIR, args.pattern, args.process_all)

    # Generate report
    if results:
        report_file = generate_report(results)
        print(f"Report saved to {report_file}")

        # Generate email list for invalid submissions
        email_list_file = generate_email_list(results)
        print(f"Follow-up email list saved to {email_list_file}")

    # Count valid and invalid prompts
    valid_count = 0
    invalid_count = 0
    for result in results:
        for prompt in result.get('prompts', []):
            if prompt.get('is_valid', False):
                valid_count += 1
            else:
                invalid_count += 1

    print("\nPrompt extraction completed!")
    print(f"Valid prompts saved to: {VALID_PROMPTS_DIR}/")
    print(f"Invalid prompts saved to: {INVALID_PROMPTS_DIR}/")

    # Summary
    print(f"\nSummary: {len(results)} total emails processed - {valid_count} valid prompts, {invalid_count} invalid prompts")

    if valid_count > 0:
        print(f"\nYou can now run the evaluation script on the valid prompts:")
        print(f"python evaluate_prompts.py --dir {VALID_PROMPTS_DIR}")

    if invalid_count > 0:
        print(f"\nPlease check {email_list_file} for a list of submissions that need follow-up")

    if len(results) == 0:
        print("\nNo new emails were processed. All emails in the directory have already had valid prompts extracted.")
        print(f"To force processing of all emails, use the --process-all flag to process them even if valid prompts exist.")


if __name__ == "__main__":
    main()