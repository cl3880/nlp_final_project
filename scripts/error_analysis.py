# scripts/error_analysis.py
#!/usr/bin/env python3
"""
Perform error analysis on the baseline model predictions.
This script analyzes false positives and false negatives to identify patterns.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_errors(predictions_file, output_dir):
    """
    Analyze errors in model predictions.
    
    Args:
        predictions_file: CSV file with predictions
        output_dir: Directory to save analysis outputs
    """
    logger.info(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Create error categories
    df['error_type'] = 'correct'
    df.loc[(df['relevant'] == True) & (df['prediction'] == 0), 'error_type'] = 'false_negative'
    df.loc[(df['relevant'] == False) & (df['prediction'] == 1), 'error_type'] = 'false_positive'
    
    # Count error types
    error_counts = df['error_type'].value_counts()
    
    # Calculate error rates
    total = len(df)
    error_rates = {
        'accuracy': (error_counts.get('correct', 0) / total) * 100,
        'false_positive_rate': (error_counts.get('false_positive', 0) / total) * 100,
        'false_negative_rate': (error_counts.get('false_negative', 0) / total) * 100
    }
    
    logger.info(f"Accuracy: {error_rates['accuracy']:.2f}%")
    logger.info(f"False positive rate: {error_rates['false_positive_rate']:.2f}%")
    logger.info(f"False negative rate: {error_rates['false_negative_rate']:.2f}%")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='error_type', data=df)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error Type')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/error_distribution.png")
    
    # Analyze probability distribution by error type
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='probability', hue='error_type', bins=20, multiple='stack')
    plt.title('Probability Distribution by Error Type')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig(f"{output_dir}/probability_distribution.png")
    
    # Analyze common terms in false negatives
    if 'false_negative' in df['error_type'].values:
        fn_docs = df[df['error_type'] == 'false_negative']
        
        # Combine title and abstract
        fn_text = (fn_docs['title'] + ' ' + fn_docs['abstract'].fillna('')).fillna('')
        
        # Generate word cloud
        if len(fn_text) > 0:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                collocations=False
            ).generate(' '.join(fn_text))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Words in False Negatives')
            plt.savefig(f"{output_dir}/false_negative_wordcloud.png")
    
    # Analyze common terms in false positives
    if 'false_positive' in df['error_type'].values:
        fp_docs = df[df['error_type'] == 'false_positive']
        
        # Combine title and abstract
        fp_text = (fp_docs['title'] + ' ' + fp_docs['abstract'].fillna('')).fillna('')
        
        # Generate word cloud
        if len(fp_text) > 0:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                collocations=False
            ).generate(' '.join(fp_text))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Common Words in False Positives')
            plt.savefig(f"{output_dir}/false_positive_wordcloud.png")
    
    # Save error examples to CSV for manual review
    df[df['error_type'] != 'correct'].to_csv(f"{output_dir}/error_examples.csv", index=False)
    
    # Create summary report
    with open(f"{output_dir}/error_analysis_summary.txt", 'w') as f:
        f.write("==== ERROR ANALYSIS SUMMARY ====\n\n")
        f.write(f"Total examples: {total}\n")
        f.write(f"Correct predictions: {error_counts.get('correct', 0)} ({error_rates['accuracy']:.2f}%)\n")
        f.write(f"False positives: {error_counts.get('false_positive', 0)} ({error_rates['false_positive_rate']:.2f}%)\n")
        f.write(f"False negatives: {error_counts.get('false_negative', 0)} ({error_rates['false_negative_rate']:.2f}%)\n\n")
        
        f.write("Probability threshold effects:\n")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            pred_at_threshold = (df['probability'] >= threshold).astype(int)
            accuracy = (pred_at_threshold == df['relevant']).mean() * 100
            recall = df.loc[df['relevant'] == True, 'probability']
            recall_at_threshold = (recall >= threshold).mean() * 100
            f.write(f"Threshold {threshold:.1f}: Accuracy {accuracy:.2f}%, Recall {recall_at_threshold:.2f}%\n")
    
    logger.info(f"Error analysis complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction errors')
    parser.add_argument('--predictions', required=True, help='CSV file with predictions')
    parser.add_argument('--output-dir', default='outputs/error_analysis', help='Output directory')
    args = parser.parse_args()
    
    analyze_errors(args.predictions, args.output_dir)

if __name__ == "__main__":
    main()