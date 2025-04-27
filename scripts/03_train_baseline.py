# train_baseline.py

#!/usr/bin/env python3
"""
Train baseline models for systematic review classification.
Implements both logistic regression and cosine similarity baselines.
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
from sklearn.metrics import make_scorer, fbeta_score

import config
from data_utils import load_data, make_splits
from baseline import (
    make_tfidf_logreg_pipeline,
    make_tfidf_cosine_pipeline,
    plot_top_features
)
from evaluate import evaluate, find_threshold_for_recall, compare_models

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.PATHS["logs_dir"], "baseline_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train baseline models for systematic review classification')
    parser.add_argument('--data', default=os.path.join(config.PATHS["data_processed"], "data_final_cleaned.csv"), 
                      help='Path to dataset CSV')
    parser.add_argument('--output-dir', default=config.PATHS["baseline_dir"],
                      help='Directory for outputs')
    parser.add_argument('--target-recall', type=float, default=0.95, 
                      help='Target recall level (default: 0.95)')
    parser.add_argument('--cos-thresh', type=float, default=0.3,
                  help='Threshold for cosine similarity classification')
    args = parser.parse_args()
    
    models_dir = os.path.join(args.output_dir, "models")
    metrics_dir = os.path.join(args.output_dir, "metrics")
    plots_dir = os.path.join(args.output_dir, "plots")
    analysis_dir = os.path.join(args.output_dir, "analysis")
    
    for directory in [args.output_dir, models_dir, metrics_dir, plots_dir, analysis_dir]:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Starting baseline model training with data from {args.data}")
    
    df = load_data(args.data)
    train, val, test = make_splits(df, stratify=True, seed=42)
    
    logger.info(f"Dataset: {len(df)} total examples ({df['relevant'].sum()} relevant)")
    logger.info(f"Train: {len(train)} examples ({train['relevant'].sum()} relevant)")
    logger.info(f"Validation: {len(val)} examples ({val['relevant'].sum()} relevant)")
    logger.info(f"Test: {len(test)} examples ({test['relevant'].sum()} relevant)")
    
    ###################
    # Logistic Regression Baseline
    ###################
    logger.info("Training logistic regression baseline")
    logreg_model = make_tfidf_logreg_pipeline()
    logreg_model.fit(train, train['relevant'])
    
    joblib.dump(logreg_model, os.path.join(models_dir, "logreg_baseline.joblib"))
    plot_top_features(logreg_model, os.path.join(plots_dir, "logreg_top_features.png"))
    
    logger.info("Evaluating logistic regression model on validation set")
    val_preds = logreg_model.predict(val)
    val_probs = logreg_model.predict_proba(val)[:, 1]
    
    val_metrics = evaluate(
        val['relevant'].values,
        val_preds,
        val_probs,
        metrics_dir,
        'logreg_validation',
        target_recall=args.target_recall
    )
    
    logger.info("Evaluating logistic regression model on test set")
    test_preds = logreg_model.predict(test)
    test_probs = logreg_model.predict_proba(test)[:, 1]
    
    test_metrics = evaluate(
        test['relevant'].values,
        test_preds,
        test_probs,
        metrics_dir,
        'logreg_test',
        target_recall=args.target_recall
    )
    
    test_with_preds = test.copy()
    test_with_preds['prediction'] = test_preds
    test_with_preds['probability'] = test_probs
    test_with_preds['correct'] = (test_with_preds['relevant'] == test_with_preds['prediction'])
    test_with_preds.to_csv(os.path.join(analysis_dir, "logreg_predictions.csv"), index=False)
    
    ###################
    # Cosine Similarity Baseline
    ###################
    logger.info("Training cosine similarity baseline")
    cos_pipeline = make_tfidf_cosine_pipeline(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=5
    )
    cos_pipeline.fit(train, train['relevant'].values)
    
    val_probs_cos = cos_pipeline.predict_proba(val)[:, 1]

    if args.cos_thresh is None:
        thresh, achieved_recall = find_threshold_for_recall(
            val['relevant'].values,
            val_probs_cos,
            args.target_recall
        )
        logger.info(f"Auto-calculated threshold: {thresh:.4f} (achieves {achieved_recall:.4f} recall)")
    else:
        thresh = args.cos_thresh
        achieved_recall = (
            (val['relevant'].values & (val_probs_cos >- thresh)).sum() / val['relevant'].sum()
        )
        logger.info(f"Using manual threshold: {thresh:.4f} (achieves {achieved_recall:.4f} recall)")

    cos_pipeline.named_steps['cosine'].threshold = thresh
    
    joblib.dump(cos_pipeline, os.path.join(models_dir, "cosine_baseline.joblib"))
    with open(os.path.join(models_dir, "cosine_threshold.json"), 'w') as f:
        json.dump({"threshold": thresh, "achieved_recall": achieved_recall}, f)
    
    logger.info("Evaluating cosine similarity model on test set")
    test_probs_cos = cos_pipeline.predict_proba(test)[:, 1]
    test_preds_cos = cos_pipeline.predict(test)
    
    cos_metrics = evaluate(
        test['relevant'].values,
        test_preds_cos,
        test_probs_cos,
        metrics_dir,
        'cosine_test',
        target_recall=args.target_recall
    )
    
    test_with_cos_preds = test.copy()
    test_with_cos_preds['prediction'] = test_preds_cos
    test_with_cos_preds['probability'] = test_probs_cos
    test_with_cos_preds['correct'] = (test_with_cos_preds['relevant'] == test_with_cos_preds['prediction'])
    test_with_cos_preds.to_csv(os.path.join(analysis_dir, "cosine_predictions.csv"), index=False)
    
    compare_models(
        [
            ('Logistic Regression', test_metrics),
            ('Cosine Similarity', cos_metrics)
        ],
        plots_dir,
        'baseline_comparison'
    )
    
    with open(os.path.join(args.output_dir, "baseline_summary.txt"), 'w') as f:
        f.write("===== BASELINE MODELS SUMMARY =====\n\n")
        
        f.write("Dataset Information:\n")
        f.write(f"- Total records: {len(df)}\n")
        f.write(f"- Relevant documents: {df['relevant'].sum()} ({df['relevant'].mean()*100:.1f}%)\n\n")
        
        f.write("Logistic Regression Model:\n")
        f.write(f"- Test AUC: {test_metrics['roc_auc']:.4f}\n")
        f.write(f"- Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"- Test Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"- Test F1: {test_metrics['f1']:.4f}\n")
        f.write(f"- Test WSS@95: {test_metrics['wss@95']:.4f}\n\n")
        
        f.write("Cosine Similarity Model:\n")
        f.write(f"- Threshold: {thresh:.4f}\n")
        f.write(f"- Test AUC: {cos_metrics['roc_auc']:.4f}\n")
        f.write(f"- Test Precision: {cos_metrics['precision']:.4f}\n")
        f.write(f"- Test Recall: {cos_metrics['recall']:.4f}\n")
        f.write(f"- Test F1: {cos_metrics['f1']:.4f}\n")
        f.write(f"- Test WSS@95: {cos_metrics['wss@95']:.4f}\n\n")
        
        if test_metrics['f1'] > cos_metrics['f1']:
            f.write("The Logistic Regression model performs better on F1 score.\n")
        elif cos_metrics['f1'] > test_metrics['f1']:
            f.write("The Cosine Similarity model performs better on F1 score.\n")
        else:
            f.write("Both models perform equally on F1 score.\n")
        
        if test_metrics['wss@95'] > cos_metrics['wss@95']:
            f.write("The Logistic Regression model saves more work at 95% recall.\n")
        elif cos_metrics['wss@95'] > test_metrics['wss@95']:
            f.write("The Cosine Similarity model saves more work at 95% recall.\n")
        else:
            f.write("Both models save the same amount of work at 95% recall.\n")
    
    logger.info(f"Baseline models training complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()

# import sys
# import os
# import argparse
# import logging
# import config
# from pathlib import Path
# import json
# import joblib
# from datetime import datetime
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
# from sklearn.metrics import make_scorer, fbeta_score

# from data_utils import load_data, make_splits
# from baseline import (
#     make_tfidf_logreg_pipeline, 
#     baseline_param_grid, 
#     plot_top_features,
#     make_tfidf_cosine_pipeline
# )
# from evaluate import evaluate, compare_models

# os.makedirs(config.PATHS["logs_dir"], exist_ok=True)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(os.path.join(config.PATHS["logs_dir"], "baseline_training.log")),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# def run_grid_search(X, y, param_grid, cv=5, scoring=None):
#     """
#     Run grid search CV to find the best model parameters.
    
#     Args:
#         X: Training features
#         y: Target labels
#         param_grid: Dictionary of parameters to search
#         cv: Number of cross-validation folds (default: 5)
#         scoring: Scoring metric (default: F2 score)
        
#     Returns:
#         Tuple of (best_model, best_params, cv_results)
#     """
#     logger.info("Starting grid search")
    
#     # Default to F2 scoring if none provided
#     if scoring is None:
#         scoring = make_scorer(fbeta_score, beta=2)
    
#     # Create baseline pipeline
#     pipeline = make_tfidf_logreg_pipeline()
    
#     # Set up cross-validation
#     cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
#     # Create and run grid search
#     grid_search = GridSearchCV(
#         pipeline,
#         param_grid,
#         scoring=scoring,
#         cv=cv_splitter,
#         n_jobs=-1,
#         verbose=1,
#         return_train_score=True
#     )
    
#     logger.info(f"Grid search with {len(param_grid)} parameters")
#     grid_search.fit(X, y)
    
#     logger.info(f"Best score: {grid_search.best_score_:.4f}")
#     logger.info(f"Best parameters: {grid_search.best_params_}")
    
#     return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

# def main():
#     os.makedirs("logs", exist_ok=True)
#     parser = argparse.ArgumentParser(description='Train baseline model for systematic review classification')
#     parser.add_argument('--data', default=os.path.join(config.PATHS["data_processed"], "data_final.csv"), help='Path to dataset CSV')
    
#     # Update to use the baseline directory from config
#     parser.add_argument(
#         '--output-dir',
#         default=config.PATHS["baseline_dir"],
#         help='Directory for all baseline artifacts'
#     )
    
#     parser.add_argument('--no-grid-search', action='store_true', help='Skip grid search and use default parameters')
#     parser.add_argument('--target-recall', type=float, default=0.95, help='Target recall level (default: 0.95)')
#     parser.add_argument('--test-size', type=float, default=0.1, help='Proportion of data to use for testing')
#     parser.add_argument('--val-size', type=float, default=0.1, help='Proportion of data to use for validation')
#     parser.add_argument('--cv-folds', type=int, default=5, help='Number of cross-validation folds')
#     args = parser.parse_args()
    
#     # Create subdirectories for baseline results
#     models_dir = os.path.join(args.output_dir, "models")
#     metrics_dir = os.path.join(args.output_dir, "metrics")
#     plots_dir = os.path.join(args.output_dir, "plots")
#     analysis_dir = os.path.join(args.output_dir, "analysis")
    
#     for directory in [args.output_dir, models_dir, metrics_dir, plots_dir, analysis_dir]:
#         os.makedirs(directory, exist_ok=True)
    
#     os.makedirs(config.PATHS["logs_dir"], exist_ok=True)
    
#     logger.info(f"Starting baseline model training with data from {args.data}")
    
#     df = load_data(args.data)
#     train, val, test = make_splits(
#         df, 
#         test_size=args.test_size, 
#         val_size=args.val_size,
#         stratify=True, 
#         seed=42
#     )
    
#     logger.info(f"Dataset: {len(df)} total examples ({df['relevant'].sum()} relevant)")
#     logger.info(f"Train: {len(train)} examples ({train['relevant'].sum()} relevant)")
#     logger.info(f"Validation: {len(val)} examples ({val['relevant'].sum()} relevant)")
#     logger.info(f"Test: {len(test)} examples ({test['relevant'].sum()} relevant)")
    
#     if args.no_grid_search:
#         logger.info("Training baseline model with default parameters")
#         baseline_model = make_tfidf_logreg_pipeline()
#         baseline_model.fit(train, train['relevant'])
#         baseline_params = {
#             'tfidf__max_features': 10000,
#             'tfidf__ngram_range': (1, 2),
#             'tfidf__min_df': 3,
#             'clf__C': 1.0,
#             'clf__class_weight': 'balanced',
#             'clf__penalty': 'l2',
#             'clf__solver': 'liblinear'
#         }
#     else:
#         logger.info("Running grid search for baseline model")
#         baseline_model, baseline_params, cv_results = run_grid_search(
#             train, 
#             train['relevant'],
#             baseline_param_grid(),
#             cv=args.cv_folds
#         )
        
#         pd.DataFrame(cv_results).to_csv(f"{metrics_dir}/grid_search_results.csv", index=False)
    
#     joblib.dump(baseline_model, f"{models_dir}/baseline_model.joblib")
#     with open(f"{models_dir}/baseline_params.json", 'w') as f:
#         json.dump(baseline_params, f, indent=2)
    
#     plot_top_features(baseline_model, f"{plots_dir}/top_features.png")
    
#     logger.info("Evaluating baseline model on validation set")
#     val_preds = baseline_model.predict(val)
#     val_probs = baseline_model.predict_proba(val)[:, 1]
    
#     val_metrics = evaluate(
#         val['relevant'].values,
#         val_preds,
#         val_probs,
#         metrics_dir,
#         'baseline_validation',
#         target_recall=args.target_recall
#     )
    
#     logger.info("Performing cross-validation")
#     cv_splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
#     cv_scoring = {
#         'f1': 'f1',
#         'precision': 'precision',
#         'recall': 'recall',
#         'roc_auc': 'roc_auc'
#     }
    
#     cv_results = cross_validate(
#         baseline_model,
#         train,
#         train['relevant'],
#         cv=cv_splitter,
#         scoring=cv_scoring,
#         return_train_score=True
#     )
    
#     for metric in cv_scoring.keys():
#         test_mean = cv_results[f'test_{metric}'].mean()
#         test_std = cv_results[f'test_{metric}'].std()
#         logger.info(f"CV {metric}: {test_mean:.4f} (±{test_std:.4f})")
    
#     cv_summary = {
#         'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#         'parameters': baseline_params
#     }
    
#     for metric in cv_scoring.keys():
#         cv_summary[f'{metric}_mean'] = float(cv_results[f'test_{metric}'].mean())
#         cv_summary[f'{metric}_std'] = float(cv_results[f'test_{metric}'].std())
    
#     with open(f"{metrics_dir}/cv_results.json", 'w') as f:
#         json.dump(cv_summary, f, indent=2)
    
#     logger.info("Evaluating baseline model on test set")
#     test_preds = baseline_model.predict(test)
#     test_probs = baseline_model.predict_proba(test)[:, 1]
    
#     test_metrics = evaluate(
#         test['relevant'].values,
#         test_preds,
#         test_probs,
#         metrics_dir,
#         'baseline_test',
#         target_recall=args.target_recall
#     )

#     logger.info("Training cosine‐similarity baseline")
#     cos_pipeline = make_tfidf_cosine_pipeline(
#         max_features=baseline_params.get('tfidf__max_features', 10000),
#         ngram_range=baseline_params.get('tfidf__ngram_range', (1,2)),
#         min_df=baseline_params.get('tfidf__min_df', 3)
#     )
#     cos_pipeline.fit(train, train['relevant'].values)

#     val_probs = cos_pipeline.predict_proba(val)[:, 1]
#     from evaluate import find_threshold_for_recall
#     thresh, _ = find_threshold_for_recall(
#         val['relevant'].values,
#         val_probs,
#         args.target_recall)
#     cos_pipeline.named_steps['cosine'].threshold = thresh

#     cos_probs = cos_pipeline.predict_proba(test)[:, 1]
#     cos_preds = cos_pipeline.predict(test)

#     evaluate(
#         test['relevant'].values,
#         cos_preds,
#         cos_probs,
#         metrics_dir,
#         'cosine_baseline',
#         target_recall=args.target_recall
#     )

#     logger.info("Cosine‐similarity baseline complete.")
    
#     test_with_preds = test.copy()
#     test_with_preds['prediction'] = test_preds
#     test_with_preds['probability'] = test_probs
#     test_with_preds['correct'] = (test_with_preds['relevant'] == test_with_preds['prediction'])
    
#     errors = test_with_preds[~test_with_preds['correct']]
#     errors.to_csv(f"{analysis_dir}/error_analysis.csv", index=False)
    
#     test_with_preds.to_csv(f"{analysis_dir}/all_predictions.csv", index=False)
    
#     logger.info("\n===== BASELINE MODEL RESULTS =====")
#     logger.info(f"Validation - AUC: {val_metrics['roc_auc']:.4f}, F1: {val_metrics['f1']:.4f}, WSS@95: {val_metrics['wss@95']:.4f}")
#     logger.info(f"Test - AUC: {test_metrics['roc_auc']:.4f}, F1: {test_metrics['f1']:.4f}, WSS@95: {test_metrics['wss@95']:.4f}")
    
#     with open(f"{args.output_dir}/summary.txt", 'w') as f:
#         f.write("===== BASELINE MODEL SUMMARY =====\n\n")
#         f.write(f"Dataset: {args.data}\n")
#         f.write(f"Total records: {len(df)}\n")
#         f.write(f"Relevant documents: {df['relevant'].sum()} ({df['relevant'].mean()*100:.1f}%)\n\n")
        
#         f.write("Data splits:\n")
#         f.write(f"- Train: {len(train)} examples ({train['relevant'].sum()} relevant)\n")
#         f.write(f"- Validation: {len(val)} examples ({val['relevant'].sum()} relevant)\n")
#         f.write(f"- Test: {len(test)} examples ({test['relevant'].sum()} relevant)\n\n")
        
#         f.write("Model parameters:\n")
#         for param, value in baseline_params.items():
#             f.write(f"- {param}: {value}\n")
#         f.write("\n")
        
#         f.write("Cross-validation results:\n")
#         for metric in cv_scoring.keys():
#             mean = cv_results[f'test_{metric}'].mean()
#             std = cv_results[f'test_{metric}'].std()
#             f.write(f"- {metric}: {mean:.4f} (±{std:.4f})\n")
#         f.write("\n")
        
#         f.write("Validation results:\n")
#         for metric, value in val_metrics.items():
#             if isinstance(value, (int, float)):
#                 f.write(f"- {metric}: {value:.4f}\n")
#         f.write("\n")
        
#         f.write("Test results:\n")
#         for metric, value in test_metrics.items():
#             if isinstance(value, (int, float)):
#                 f.write(f"- {metric}: {value:.4f}\n")
#         f.write("\n")
        
#         f.write("Error analysis:\n")
#         f.write(f"- False positives: {sum((test_preds == 1) & (test['relevant'] == 0))}\n")
#         f.write(f"- False negatives: {sum((test_preds == 0) & (test['relevant'] == 1))}\n")
#         f.write(f"- Error analysis file: analysis/error_analysis.csv\n")
    
#     logger.info(f"Results saved to {args.output_dir}")
#     logger.info("Baseline model training complete.")
    
# if __name__ == "__main__":
#     main()