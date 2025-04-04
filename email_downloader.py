#!/usr/bin/env python3
"""
Email Downloader

This script:
1. Fetches emails from Gmail with a specific label
2. Saves the complete email body to individual files
3. Creates a metadata file with email information for later processing
"""

import argparse
import datetime
import email
import glob
import imaplib
import json
import os
import re
from email.header import decode_header

# Configure email settings for fetching submissions
EMAIL_ADDRESS = "email address"  # Replace with your email address
EMAIL_PASSWORD = "app password"  # Use app password for Gmail
MAX_EMAILS = 200  # Maximum number of emails to process

# Output directory
EMAILS_DIR = "raw_emails"


def load_existing_index(index_file):
    """Load existing email index file to track previously downloaded emails

    Args:
        index_file: Path to existing index file

    Returns:
        Set of email IDs already downloaded
    """
    existing_ids = set()

    if os.path.exists(index_file):
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

                # Extract all email IDs from the index
                for email_data in index_data.get('emails', []):
                    if 'email_id' in email_data:
                        existing_ids.add(email_data['email_id'])

                print(f"Loaded {len(existing_ids)} existing email IDs from {index_file}")
        except Exception as e:
            print(f"Error loading existing index file: {e}")

    return existing_ids


def setup_directories():
    """Create output directories if they don't exist"""
    os.makedirs(EMAILS_DIR, exist_ok=True)
    print(f"Created directory: {EMAILS_DIR}")


def parse_email_metadata(email_message):
    """Parse email header information

    Args:
        email_message: Email message object

    Returns:
        Tuple of (sender_name, email_address, subject, email_timestamp)
    """
    # Extract subject
    subject = email_message.get('Subject', '')
    subject_decoded = decode_header(subject)[0][0]
    if isinstance(subject_decoded, bytes):
        subject = subject_decoded.decode('utf-8', errors='replace')
    else:
        subject = str(subject_decoded)

    # Extract sender information
    from_field = email_message.get('From', 'Unknown')

    # Try to extract name and email
    name_match = re.search(r"^(.*?)\s*<(.+?)>", from_field)
    if name_match:
        sender_name = name_match.group(1).strip()
        sender_email = name_match.group(2).strip()
    else:
        sender_name = from_field.strip()
        sender_email = from_field.strip()

    # Get email timestamp
    date = email_message.get('Date')
    email_timestamp = email.utils.parsedate_to_datetime(
        date).isoformat() if date else datetime.datetime.now().isoformat()

    return sender_name, sender_email, subject, email_timestamp


def extract_email_body(email_message):
    """Extract the body text from an email

    Args:
        email_message: Email message object

    Returns:
        String containing the email body
    """
    body = ""

    # Walk through all parts of the message
    if email_message.is_multipart():
        print("Processing multipart message...")
        for part in email_message.walk():
            if part.get_content_type() == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or 'utf-8'
                        part_text = payload.decode(charset, errors='replace')
                        body += part_text + "\n"
                        print(f"Added text part with {len(part_text)} characters")
                except Exception as e:
                    print(f"Error processing text part: {e}")
    else:
        try:
            payload = email_message.get_payload(decode=True)
            if payload:
                charset = email_message.get_content_charset() or 'utf-8'
                body = payload.decode(charset, errors='replace')
                print(f"Added non-multipart text with {len(body)} characters")
        except Exception as e:
            print(f"Error decoding non-multipart body: {e}")

    return body


def save_email(sender_name, sender_email, subject, body, email_timestamp, email_id):
    """Save email body and metadata to files

    Args:
        sender_name: Name of the email sender
        sender_email: Email address of the sender
        subject: Email subject
        body: Full email body
        email_timestamp: Timestamp of the email
        email_id: Unique email ID

    Returns:
        Tuple of (body_filepath, metadata_filepath)
    """
    # Sanitize name for filename
    safe_name = re.sub(r'[^\w\s-]', '', sender_name).strip().replace(' ', '_')

    # Create a unique filename with timestamp and email ID
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Email body file
    body_filename = f"{EMAILS_DIR}/{safe_name}_{timestamp}_{email_id}.txt"

    # Metadata file
    metadata_filename = f"{EMAILS_DIR}/{safe_name}_{timestamp}_{email_id}_metadata.json"

    # Save the email body
    with open(body_filename, 'w', encoding='utf-8') as f:
        f.write(body)

    # Create and save metadata
    metadata = {
        'sender_name': sender_name,
        'sender_email': sender_email,
        'subject': subject,
        'email_timestamp': email_timestamp,
        'email_id': email_id,
        'body_file': body_filename,
        'download_timestamp': datetime.datetime.now().isoformat()
    }

    with open(metadata_filename, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved email body to {body_filename}")
    print(f"Saved email metadata to {metadata_filename}")

    return body_filename, metadata_filename


def fetch_emails(email_address, password, max_emails=100, existing_index=None):
    """Fetch emails from Gmail and save them

    Args:
        email_address: Email address to log in with
        password: Email password or app password
        max_emails: Maximum number of emails to process
        existing_index: Path to existing index file to check for already downloaded emails

    Returns:
        List of saved email metadata
    """
    saved_emails = []
    mail = None

    # Load existing email IDs if an index file is provided
    existing_ids = set()
    if existing_index:
        existing_ids = load_existing_index(existing_index)

    try:
        # Connect to Gmail
        print("Connecting to Gmail...")
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(email_address, password)

        # First, try to select the specific label
        print("Selecting the 'PromptHomework' label...")
        status, messages = mail.select("PromptHomework")
        print(f"Select status: {status}, message count: {messages[0].decode() if status == 'OK' else 'N/A'}")

        # If label selection fails, fall back to search criteria
        if status != 'OK':
            print("Could not select the PromptHomework label. Falling back to inbox...")
            status, messages = mail.select("INBOX")
            print(f"Select inbox status: {status}, message count: {messages[0].decode() if status == 'OK' else 'N/A'}")

            if status != 'OK':
                print("Could not select any mailbox. Exiting...")
                return saved_emails

        # Search for all messages in the selected mailbox/label
        print("Searching for all messages in the selected mailbox...")
        status, email_ids = mail.search(None, 'ALL')

        if status != 'OK' or not email_ids[0]:
            print("Search command returned no emails")
            return saved_emails

        email_id_list = email_ids[0].split()
        print(f"Found {len(email_id_list)} emails in the selected mailbox")

        # Process in reverse order (newest first), up to max_emails
        email_id_list = list(reversed(email_id_list))

        # Count how many new emails we'll process
        new_emails = [eid.decode() for eid in email_id_list if eid.decode() not in existing_ids]
        new_emails = new_emails[:max_emails]  # Only process up to max_emails

        print(f"Found {len(new_emails)} new emails to download")

        for i, email_id_str in enumerate(new_emails, 1):
            print(f"\nProcessing email {i}/{len(new_emails)} (ID: {email_id_str})")

            # Fetch the email
            status, msg_data = mail.fetch(email_id_str.encode(), "(RFC822)")
            if status != 'OK':
                print(f"Failed to fetch email with ID: {email_id_str}")
                continue

            email_message = email.message_from_bytes(msg_data[0][1])

            # Parse the email metadata
            sender_name, sender_email, subject, email_timestamp = parse_email_metadata(email_message)

            # Extract the email body
            body = extract_email_body(email_message)

            # Save the email body and metadata
            body_file, metadata_file = save_email(
                sender_name, sender_email, subject, body, email_timestamp, email_id_str
            )

            # Add to saved emails list
            email_data = {
                'sender_name': sender_name,
                'sender_email': sender_email,
                'subject': subject,
                'email_timestamp': email_timestamp,
                'email_id': email_id_str,
                'body_file': body_file,
                'metadata_file': metadata_file
            }
            saved_emails.append(email_data)

            print(f"Successfully saved email from {sender_name}")

        if not new_emails:
            print("No new emails to download")

    except Exception as e:
        print(f"Error fetching emails: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Properly close the connection
        if mail is not None:
            try:
                mail.close()
            except:
                pass
            try:
                mail.logout()
            except:
                pass
            print("Email connection closed")

    return saved_emails


def generate_index(saved_emails, existing_index_file=None):
    """Generate or update an index file with all saved emails

    Args:
        saved_emails: List of newly saved email metadata
        existing_index_file: Path to existing index file (if any)

    Returns:
        Path to the index file
    """
    # Determine the index filename
    if existing_index_file and os.path.exists(existing_index_file):
        index_file = existing_index_file

        # Load existing index data
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

                # Append new emails to existing ones
                existing_emails = index_data.get('emails', [])
                all_emails = existing_emails + saved_emails

                # Update the index data
                index_data['download_timestamp'] = datetime.datetime.now().isoformat()
                index_data['email_count'] = len(all_emails)
                index_data['emails'] = all_emails

                print(f"Updating existing index with {len(saved_emails)} new emails")
        except Exception as e:
            print(f"Error reading existing index file: {e}")
            # Create new index if there's an issue with the existing one
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            index_file = f"email_index_{timestamp}.json"
            index_data = {
                'download_timestamp': datetime.datetime.now().isoformat(),
                'email_count': len(saved_emails),
                'emails': saved_emails
            }
    else:
        # Create a new index file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        index_file = f"email_index_{timestamp}.json"
        index_data = {
            'download_timestamp': datetime.datetime.now().isoformat(),
            'email_count': len(saved_emails),
            'emails': saved_emails
        }

    # Write index to file
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)

    print(f"Index file saved to {index_file}")
    return index_file


def main():
    global EMAIL_ADDRESS, EMAIL_PASSWORD

    parser = argparse.ArgumentParser(description="Download emails and save bodies")
    parser.add_argument("--email", type=str, default=None, help="Gmail address to use")
    parser.add_argument("--password", type=str, default=None, help="Gmail app password")
    parser.add_argument("--max-emails", type=int, default=MAX_EMAILS, help="Maximum number of emails to process")
    parser.add_argument("--index", type=str, default=None,
                        help="Path to existing index file to avoid re-downloading emails")
    args = parser.parse_args()

    # Set up credentials - try command line args, then environment variables, then interactive input
    EMAIL_ADDRESS = args.email or os.environ.get("GMAIL_ADDRESS") or EMAIL_ADDRESS
    EMAIL_PASSWORD = args.password or os.environ.get("GMAIL_PASSWORD") or EMAIL_PASSWORD

    # Ask for missing credentials interactively
    if EMAIL_ADDRESS == "your_email@example.com" or not EMAIL_ADDRESS:
        EMAIL_ADDRESS = input("Enter your Gmail address: ")

    if EMAIL_PASSWORD == "your_app_password" or not EMAIL_PASSWORD:
        import getpass
        EMAIL_PASSWORD = getpass.getpass("Enter your Gmail app password: ")

    # Create output directories
    setup_directories()

    # Fetch and save emails
    print(f"Starting email download from {EMAIL_ADDRESS}...")

    # Find the latest index file if none specified
    if not args.index:
        index_files = sorted(glob.glob("email_index_*.json"), reverse=True)
        if index_files:
            latest_index = index_files[0]
            print(f"Found latest index file: {latest_index}")
            args.index = latest_index

    # Fetch emails, checking against existing index if provided
    saved_emails = fetch_emails(EMAIL_ADDRESS, EMAIL_PASSWORD, args.max_emails, args.index)

    # Generate or update index file
    if saved_emails:
        index_file = generate_index(saved_emails, args.index)
        print(f"Index file saved to {index_file}")

    print("\nEmail download completed!")
    print(f"Downloaded {len(saved_emails)} new emails to {EMAILS_DIR}/")

    if saved_emails:
        print("\nYou can now run the prompt extractor on these saved emails.")
    else:
        print("\nNo new emails were downloaded.")

    # Return the index file in case it's needed for prompt extraction
    if args.index or saved_emails:
        index_to_use = args.index if args.index and not saved_emails else (index_file if saved_emails else None)
        if index_to_use:
            print(f"\nUse this index file for prompt extraction:")
            print(f"python prompt-extractor.py --dir {EMAILS_DIR} --index {index_to_use}")


if __name__ == "__main__":
    main()
