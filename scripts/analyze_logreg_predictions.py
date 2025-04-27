#!/usr/bin/env python3
"""
Analyze logistic regression prediction errors to understand patterns.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
import config

# Configure output paths
OUTPUT_DIR = os.path.join(config.PATHS["baseline_dir"], "error_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_key_terms(text, lowercase=True):
    """Extract important terms from text."""
    if pd.isna(text) or not isinstance(text, str):
        return []
        
    if lowercase:
        text = text.lower()
        
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stopwords = set(['a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                    'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 
                    'by', 'about', 'of', 'from', 'as', 'this', 'that', 'these', 
                    'those', 'it', 'its', 'it\'s', 'they', 'them', 'their'])
    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    
    return tokens

def check_key_terms(text, terms, lowercase=True):
    """Check if text contains any of the terms."""
    if pd.isna(text) or not isinstance(text, str):
        return False
        
    if lowercase:
        text = text.lower()
        
    return any(term.lower() in text.lower() for term in terms)

def analyze_predictions(predictions_file, output_dir=OUTPUT_DIR):
    """Analyze prediction errors to identify patterns."""
    
    print(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Overall statistics
    correct = df[df['correct']]
    incorrect = df[~df['correct']]
    
    # Further separate into false positives and false negatives
    false_positives = df[(df['prediction'] == True) & (df['relevant'] == False)]
    false_negatives = df[(df['prediction'] == False) & (df['relevant'] == True)]
    
    print(f"Total test examples: {len(df)}")
    print(f"Correct predictions: {len(correct)} ({len(correct)/len(df)*100:.1f}%)")
    print(f"False positives: {len(false_positives)} ({len(false_positives)/len(df)*100:.1f}%)")
    print(f"False negatives: {len(false_negatives)} ({len(false_negatives)/len(df)*100:.1f}%)")
    
    # Probability distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df[df['relevant']==True]['probability'], 
                 label='True Relevant', alpha=0.6, bins=20, color='green')
    sns.histplot(df[df['relevant']==False]['probability'], 
                 label='True Irrelevant', alpha=0.6, bins=20, color='red')
    plt.axvline(0.5, color='black', linestyle='--', label='Decision threshold')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Probabilities')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()
    
    # Define key terms based on manual inspection of features
    relevant_terms = ['obliteration', 'rate', 'radiosurg', 'gamma knife', 'avm', 
                       'arteriovenous malformation', 'srs', 'nidus', 'stereotactic', 
                       'occlusion', 'cyberknife']
    
    irrelevant_terms = ['pediatric', 'children', 'meta analysis', 'systematic review', 
                        'case report', 'surgery', 'surgical', 'resection']
    
    # Check for key terms in false predictions
    fn_with_relevant_terms = false_negatives[
        false_negatives['title'].apply(lambda x: check_key_terms(x, relevant_terms)) | 
        false_negatives['abstract'].apply(lambda x: check_key_terms(x, relevant_terms))
    ]
    
    fp_with_irrelevant_terms = false_positives[
        false_positives['title'].apply(lambda x: check_key_terms(x, irrelevant_terms)) | 
        false_positives['abstract'].apply(lambda x: check_key_terms(x, irrelevant_terms))
    ]
    
    print(f"\nFalse negatives containing relevant terms: {len(fn_with_relevant_terms)} "
          f"({len(fn_with_relevant_terms)/len(false_negatives)*100:.1f}%)")
    print(f"False positives containing irrelevant terms: {len(fp_with_irrelevant_terms)} "
          f"({len(fp_with_irrelevant_terms)/len(false_positives)*100:.1f}%)")
    
    # Extract common words from false predictions
    fp_words = []
    for _, row in false_positives.iterrows():
        title_words = extract_key_terms(row['title'])
        abstract_words = extract_key_terms(row['abstract'])
        fp_words.extend(title_words + abstract_words)
    
    fn_words = []
    for _, row in false_negatives.iterrows():
        title_words = extract_key_terms(row['title'])
        abstract_words = extract_key_terms(row['abstract'])
        fn_words.extend(title_words + abstract_words)
    
    # Count word frequencies
    fp_word_counts = Counter(fp_words)
    fn_word_counts = Counter(fn_words)
    
    # Plot top words
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    top_fp_words = [word for word, count in fp_word_counts.most_common(15)]
    fp_counts = [count for word, count in fp_word_counts.most_common(15)]
    plt.barh(top_fp_words, fp_counts, color='red', alpha=0.7)
    plt.title('Top Words in False Positives')
    plt.xlabel('Frequency')
    
    plt.subplot(1, 2, 2)
    if fn_words:  # Only if there are false negatives
        top_fn_words = [word for word, count in fn_word_counts.most_common(min(15, len(fn_word_counts)))]
        fn_counts = [count for word, count in fn_word_counts.most_common(min(15, len(fn_word_counts)))]
        plt.barh(top_fn_words, fn_counts, color='blue', alpha=0.7)
        plt.title('Top Words in False Negatives')
        plt.xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_word_frequencies.png'))
    plt.close()
    
    # Save detailed error examples for manual review
    false_positives.to_csv(os.path.join(output_dir, 'false_positives.csv'), index=False)
    false_negatives.to_csv(os.path.join(output_dir, 'false_negatives.csv'), index=False)
    
    # Generate error analysis report
    with open(os.path.join(output_dir, 'error_analysis_report.txt'), 'w') as f:
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("====================\n\n")
        
        f.write(f"Total test examples: {len(df)}\n")
        f.write(f"Correct predictions: {len(correct)} ({len(correct)/len(df)*100:.1f}%)\n")
        f.write(f"Incorrect predictions: {len(incorrect)} ({len(incorrect)/len(df)*100:.1f}%)\n\n")
        
        f.write(f"False positives: {len(false_positives)} ({len(false_positives)/len(df)*100:.1f}%)\n")
        f.write(f"False negatives: {len(false_negatives)} ({len(false_negatives)/len(df)*100:.1f}%)\n\n")
        
        f.write("False Positive Analysis:\n")
        f.write(f"- {len(fp_with_irrelevant_terms)} out of {len(false_positives)} "
                f"({len(fp_with_irrelevant_terms)/len(false_positives)*100:.1f}%) "
                f"contain explicit irrelevant terms\n")
        f.write("- Most common words in false positives:\n")
        for word, count in fp_word_counts.most_common(10):
            f.write(f"  * {word}: {count}\n")
        
        f.write("\nFalse Negative Analysis:\n")
        f.write(f"- {len(fn_with_relevant_terms)} out of {len(false_negatives)} "
                f"({len(fn_with_relevant_terms)/len(false_negatives)*100:.1f}%) "
                f"contain explicit relevant terms\n")
        if fn_words:
            f.write("- Most common words in false negatives:\n")
            for word, count in fn_word_counts.most_common(min(10, len(fn_word_counts))):
                f.write(f"  * {word}: {count}\n")
        
        f.write("\nConclusions:\n")
        if len(false_negatives) == 0:
            f.write("- The model is achieving excellent recall, with no false negatives.\n")
        elif len(fn_with_relevant_terms) / len(false_negatives) > 0.5:
            f.write("- Most false negatives contain relevant terms but are still missed.\n"
                    "  This suggests context or specific combinations of terms matter.\n")
        
        if len(fp_with_irrelevant_terms) / len(false_positives) < 0.5:
            f.write("- Most false positives don't contain obvious irrelevant terms.\n"
                    "  This suggests they have features that strongly resemble relevant documents.\n")
        else:
            f.write("- Many false positives contain terms that should mark them as irrelevant.\n"
                    "  This suggests the model needs to weight these terms more heavily.\n")
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return df, false_positives, false_negatives

def main():
    parser = argparse.ArgumentParser(description='Analyze logistic regression prediction errors')
    parser.add_argument('--predictions', 
                      default=os.path.join(config.PATHS["baseline_dir"], "analysis/logreg_predictions.csv"),
                      help='Path to predictions CSV file')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                      help='Directory for output files')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    analyze_predictions(args.predictions, args.output_dir)

if __name__ == "__main__":
    main()