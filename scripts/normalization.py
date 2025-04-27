#!/usr/bin/env python3
"""
Compare baseline model with different text normalization approaches.
"""
import os
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import config
from data_utils import load_data, make_splits, combine_text_fields
from baseline import (
    make_tfidf_logreg_pipeline,
    make_tfidf_cosine_pipeline,
    TextCombiner,
    plot_top_features
)
from evaluate import evaluate, find_threshold_for_recall, compare_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.PATHS["logs_dir"], "normalization_comparison.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TextNormalizer:
    """Utility class for text normalization."""
    
    @staticmethod
    def normalize(text, stemming=False, lemmatization=False):
        """Normalize text by applying stemming or lemmatization."""
        import re
        from nltk.tokenize import word_tokenize
        
        if pd.isna(text) or text is None or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except hyphens (to preserve hyphenated terms)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Tokenize
        try:
            import nltk
            tokens = word_tokenize(text)
        except:
            # Fallback if NLTK not available
            tokens = text.split()
            
        normalized_tokens = []
        
        if stemming:
            try:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                normalized_tokens = [stemmer.stem(token) for token in tokens]
            except:
                logger.warning("Stemming failed, NLTK may not be properly installed.")
                normalized_tokens = tokens
        elif lemmatization:
            try:
                from nltk.stem import WordNetLemmatizer
                import nltk
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    nltk.download('wordnet')
                    nltk.download('punkt')
                lemmatizer = WordNetLemmatizer()
                normalized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
            except:
                logger.warning("Lemmatization failed, NLTK may not be properly installed.")
                normalized_tokens = tokens
        else:
            normalized_tokens = tokens
        
        # Rejoin
        return ' '.join(normalized_tokens)

class NormalizingTextCombiner(TextCombiner):
    """Combines text columns and applies normalization."""
    
    def __init__(self, text_columns=['title', 'abstract'], use_stemming=False, use_lemmatization=False):
        super().__init__(text_columns)
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
    def transform(self, X):
        """Combine text columns and apply normalization."""
        # First combine texts using parent method
        combined_texts = super().transform(X)
        
        # Apply normalization
        normalized_texts = [
            TextNormalizer.normalize(
                text, 
                stemming=self.use_stemming, 
                lemmatization=self.use_lemmatization
            ) 
            for text in combined_texts
        ]
        
        return normalized_texts

def make_normalized_pipeline(max_features=10000, 
                            ngram_range=(1, 2),
                            min_df=3,
                            text_columns=['title', 'abstract'],
                            C=1.0,
                            class_weight='balanced',
                            use_stemming=False,
                            use_lemmatization=False):
    """Create a pipeline with text normalization."""
    
    normalizer_name = "stemming" if use_stemming else ("lemmatization" if use_lemmatization else "none")
    logger.info(f"Creating TF-IDF + LogReg pipeline with {normalizer_name} normalization")
    
    return Pipeline([
        ('combiner', NormalizingTextCombiner(
            text_columns, 
            use_stemming=use_stemming, 
            use_lemmatization=use_lemmatization
        )),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            sublinear_tf=True,
        )),
        ('clf', LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=5000,
            solver='liblinear',
            random_state=42
        ))
    ])

def main():
    parser = argparse.ArgumentParser(description='Compare text normalization approaches')
    parser.add_argument('--data', default=os.path.join(config.PATHS["data_processed"], "data_final.csv"), 
                      help='Path to dataset CSV')
    parser.add_argument('--output-dir', default=os.path.join(config.PATHS["results_dir"], "normalization"),
                      help='Directory for outputs')
    parser.add_argument('--target-recall', type=float, default=0.95, 
                      help='Target recall level (default: 0.95)')
    args = parser.parse_args()
    
    # Create output directories
    models_dir = os.path.join(args.output_dir, "models")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    plots_dir = os.path.join(args.output_dir, "plots")
    analysis_dir = os.path.join(args.output_dir, "analysis")
    
    for directory in [args.output_dir, models_dir, metrics_dir, plots_dir, analysis_dir]:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Starting normalization comparison with data from {args.data}")
    
    # Load and split data
    df = load_data(args.data)
    train, val, test = make_splits(df, stratify=True, seed=42)
    
    logger.info(f"Dataset: {len(df)} total examples ({df['relevant'].sum()} relevant)")
    logger.info(f"Train: {len(train)} examples ({train['relevant'].sum()} relevant)")
    logger.info(f"Validation: {len(val)} examples ({val['relevant'].sum()} relevant)")
    logger.info(f"Test: {len(test)} examples ({test['relevant'].sum()} relevant)")
    
    # Get the best hyperparameters from previous baseline
    model_params = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 5,
        'C': 1.0,
        'class_weight': 'balanced'
    }
    
    # Configure models to compare
    models = {
        'baseline': make_tfidf_logreg_pipeline(
            max_features=model_params['max_features'],
            ngram_range=model_params['ngram_range'],
            min_df=model_params['min_df'],
            C=model_params['C'],
            class_weight=model_params['class_weight']
        ),
        'stemming': make_normalized_pipeline(
            max_features=model_params['max_features'],
            ngram_range=model_params['ngram_range'],
            min_df=model_params['min_df'],
            C=model_params['C'],
            class_weight=model_params['class_weight'],
            use_stemming=True
        ),
        'lemmatization': make_normalized_pipeline(
            max_features=model_params['max_features'],
            ngram_range=model_params['ngram_range'],
            min_df=model_params['min_df'],
            C=model_params['C'],
            class_weight=model_params['class_weight'],
            use_lemmatization=True
        )
    }
    
    # Train and evaluate models
    all_metrics = {}
    
    for model_name, model in models.items():
        logger.info(f"Training {model_name} model")
        model.fit(train, train['relevant'])
        
        # Save model
        joblib.dump(model, os.path.join(models_dir, f"{model_name}_model.joblib"))
        
        # Plot top features
        plot_top_features(model, os.path.join(plots_dir, f"{model_name}_top_features.png"))
        
        # Evaluate on validation set
        logger.info(f"Evaluating {model_name} model on validation set")
        val_preds = model.predict(val)
        val_probs = model.predict_proba(val)[:, 1]
        
        val_metrics = evaluate(
            val['relevant'].values,
            val_preds,
            val_probs,
            metrics_dir,
            f'{model_name}_validation',
            target_recall=args.target_recall
        )
        
        # Evaluate on test set
        logger.info(f"Evaluating {model_name} model on test set")
        test_preds = model.predict(test)
        test_probs = model.predict_proba(test)[:, 1]
        
        test_metrics = evaluate(
            test['relevant'].values,
            test_preds,
            test_probs,
            metrics_dir,
            f'{model_name}_test',
            target_recall=args.target_recall
        )
        
        all_metrics[model_name] = test_metrics
        
        # Save predictions for analysis
        test_with_preds = test.copy()
        test_with_preds['prediction'] = test_preds
        test_with_preds['probability'] = test_probs
        test_with_preds['correct'] = (test_with_preds['relevant'] == test_with_preds['prediction'])
        test_with_preds.to_csv(os.path.join(analysis_dir, f"{model_name}_predictions.csv"), index=False)
    
    # Compare models
    model_list = [(name, metrics) for name, metrics in all_metrics.items()]
    compare_models(model_list, plots_dir, 'normalization_comparison')
    
    # Create summary report
    with open(os.path.join(args.output_dir, "normalization_summary.txt"), 'w') as f:
        f.write("===== TEXT NORMALIZATION COMPARISON =====\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- Total records: {len(df)}\n")
        f.write(f"- Relevant documents: {df['relevant'].sum()} ({df['relevant'].mean()*100:.1f}%)\n\n")
        
        for model_name, metrics in all_metrics.items():
            f.write(f"{model_name.capitalize()} Model:\n")
            f.write(f"- Test AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"- Test Precision: {metrics['precision']:.4f}\n")
            f.write(f"- Test Recall: {metrics['recall']:.4f}\n")
            f.write(f"- Test F1: {metrics['f1']:.4f}\n")
            f.write(f"- Test WSS@95: {metrics['wss@95']:.4f}\n\n")
        
        # Determine best model by F1 score
        best_model = max(all_metrics.items(), key=lambda x: x[1]['f1'])[0]
        f.write(f"The {best_model.capitalize()} model performs best on F1 score.\n")
        
        # Determine best model by WSS@95
        best_wss_model = max(all_metrics.items(), key=lambda x: x[1]['wss@95'])[0]
        f.write(f"The {best_wss_model.capitalize()} model saves most work at 95% recall.\n")
    
    logger.info(f"Normalization comparison complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()