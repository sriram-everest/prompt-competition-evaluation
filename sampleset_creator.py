#!/usr/bin/env python3
"""
Fixed Sample Dataset Creator (Fixed version)

Creates a fixed sample of examples from the Amazon Polarity 30K dataset,
which will be saved and reused for all prompt evaluations to ensure
consistent and fair comparisons.
"""

import pickle

import pandas as pd
from datasets import load_dataset

# Output file
FIXED_SAMPLE_FILE = "fixed_evaluation_sample.pkl"
SAMPLE_SIZE = 200


def create_fixed_sample():
    """
    Creates a fixed sample of examples from the Amazon Polarity 30K dataset
    with a good balance of easy, medium, and challenging examples.
    """
    print("Creating fixed evaluation sample...")

    # Load the original dataset
    print("Loading original dataset...")
    dataset = load_dataset("rungalileo/amazon_polarity_30k")

    # Convert to pandas for easier handling
    df = pd.DataFrame(dataset['train'])

    # Convert labels: 0 (positive) -> "SAD", 1 (negative) -> "HAPPY" according to the twist
    df['correct_label'] = df['label'].apply(lambda x: "SAD" if x == 0 else "HAPPY")
    df = df.rename(columns={'text': 'review_text'})

    print(f"Original dataset size: {len(df)} examples")

    # Add difficulty indicators
    print("Analyzing review difficulty...")
    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()

    # Identify reviews with mixed or nuanced sentiment
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'love',
                      'awesome', 'fantastic', 'perfect', 'recommend', 'enjoyed', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'worst', 'horrible', 'poor', 'waste',
                      'disappointing', 'worse', 'unfortunately', 'boring', 'problem', 'issue']

    # Count sentiment words
    df['pos_count'] = df['review_text'].str.lower().apply(
        lambda x: sum(1 for word in positive_words if word in x.lower().split()))
    df['neg_count'] = df['review_text'].str.lower().apply(
        lambda x: sum(1 for word in negative_words if word in x.lower().split()))

    # Calculate mixed sentiment score
    df['mixed_sentiment'] = df.apply(lambda row: min(row['pos_count'], row['neg_count']), axis=1)

    # Identify reviews with negations (not, isn't, wasn't, etc.)
    negation_pattern = r'\b(?:not|no|never|don\'t|doesn\'t|isn\'t|wasn\'t|aren\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|can\'t|couldn\'t|shouldn\'t|wouldn\'t)\b'
    df['has_negation'] = df['review_text'].str.lower().str.contains(negation_pattern, regex=True)

    # Create an overall difficulty score
    df['difficulty'] = (
        # Length component: longer reviews tend to be more nuanced
            (df['word_count'] / df['word_count'].max() * 0.3) +
            # Mixed sentiment component
            (df['mixed_sentiment'] * 2 * 0.4) +
            # Negation component
            (df['has_negation'].astype(int) * 0.3)
    )

    # Normalize difficulty to 0-100 scale
    df['difficulty'] = ((df['difficulty'] - df['difficulty'].min()) /
                        (df['difficulty'].max() - df['difficulty'].min())) * 100

    # Categorize by difficulty
    df['difficulty_level'] = pd.cut(
        df['difficulty'],
        bins=[0, 33, 66, 100],
        labels=['easy', 'medium', 'hard']
    )

    print("Difficulty distribution in original dataset:")
    difficulty_counts = df['difficulty_level'].value_counts()
    for level, count in difficulty_counts.items():
        print(f"  {level}: {count} ({count / len(df):.1%})")

    # Create a stratified sample:
    # 40% easy, 40% medium, 20% hard examples
    sample_size = SAMPLE_SIZE
    easy_count = int(sample_size * 0.4)
    medium_count = int(sample_size * 0.4)
    hard_count = sample_size - easy_count - medium_count

    print(f"\nCreating stratified sample with {sample_size} examples:")
    print(f"  Easy: {easy_count} ({easy_count / sample_size:.1%})")
    print(f"  Medium: {medium_count} ({medium_count / sample_size:.1%})")
    print(f"  Hard: {hard_count} ({hard_count / sample_size:.1%})")

    # For each difficulty level, balance the classes (HAPPY/SAD)
    sample_dfs = []

    for difficulty, count in [('easy', easy_count), ('medium', medium_count), ('hard', hard_count)]:
        difficulty_df = df[df['difficulty_level'] == difficulty]
        happy_df = difficulty_df[difficulty_df['correct_label'] == 'HAPPY']
        sad_df = difficulty_df[difficulty_df['correct_label'] == 'SAD']

        # Calculate how many samples to take from each class
        happy_count = count // 2
        sad_count = count - happy_count

        # Check if we have enough samples in each class
        if len(happy_df) < happy_count:
            print(
                f"Warning: Not enough HAPPY samples in {difficulty} category. Using {len(happy_df)} instead of {happy_count}")
            happy_count = len(happy_df)
            # Adjust sad_count to maintain total count
            sad_count = min(count - happy_count, len(sad_df))

        if len(sad_df) < sad_count:
            print(
                f"Warning: Not enough SAD samples in {difficulty} category. Using {len(sad_df)} instead of {sad_count}")
            sad_count = len(sad_df)
            # Re-adjust happy_count if needed
            happy_count = min(count - sad_count, len(happy_df))

        # Take samples
        happy_sample = happy_df.sample(happy_count, random_state=42)
        sad_sample = sad_df.sample(sad_count, random_state=42)

        # Combine
        difficulty_sample = pd.concat([happy_sample, sad_sample])
        sample_dfs.append(difficulty_sample)

    # Combine all difficulty levels
    final_sample = pd.concat(sample_dfs)

    # Shuffle the sample
    final_sample = final_sample.sample(frac=1, random_state=42).reset_index(drop=True)

    # Keep only relevant columns
    final_sample = final_sample[['review_text', 'correct_label', 'difficulty_level']]

    # Save the fixed sample
    with open(FIXED_SAMPLE_FILE, 'wb') as f:
        pickle.dump(final_sample, f)

    # Print some statistics
    print("\nFixed sample created successfully!")
    print(f"Sample size: {len(final_sample)} examples")
    print("Class distribution:")
    label_counts = final_sample['correct_label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count / len(final_sample):.1%})")

    print("Difficulty distribution:")
    difficulty_counts = final_sample['difficulty_level'].value_counts()
    for level, count in difficulty_counts.items():
        print(f"  {level}: {count} ({count / len(final_sample):.1%})")

    print(f"\nFixed sample saved to {FIXED_SAMPLE_FILE}")

    # Print some example reviews from each difficulty level
    print("\nExample reviews from each difficulty level:")
    difficulty_levels = ['easy', 'medium', 'hard']

    for level in difficulty_levels:
        examples = final_sample[final_sample['difficulty_level'] == level].head(2)
        print(f"\n{level.upper()} examples:")
        for i, (_, row) in enumerate(examples.iterrows(), 1):
            print(f"  Example {i} ({row['correct_label']}):")
            print(f"  {row['review_text'][:100]}...")

    return final_sample


if __name__ == "__main__":
    create_fixed_sample()
