# utils/evaluate.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    classification_report,
    roc_auc_score,
)
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def calculate_wss_at_recall(y_true, y_scores, target_recall=0.95):
    """
    Calculate Work Saved over Sampling at a target recall level.
    
    Args:
        y_true: Array-like of true binary labels
        y_scores: Array-like of predicted scores or probabilities
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        float: Work saved over sampling metric value
    """
    # Ensure arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Sort by prediction score (highest first)
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]
    
    # Find minimum number of documents needed to achieve target recall
    n_pos = y_true.sum()
    target_pos = int(np.ceil(n_pos * target_recall))
    
    # Count until we find enough positive examples
    found_pos = 0
    for i, val in enumerate(y_sorted):
        found_pos += val
        if found_pos >= target_pos:
            n_reviewed = i + 1
            break
    else:
        # If we never reached target recall, we reviewed all documents
        n_reviewed = len(y_true)
    
    # Calculate work saved
    n_total = len(y_true)
    wss = 1.0 - (n_reviewed / n_total) - (1.0 - target_recall)
    
    return wss

def find_threshold_for_recall(y_true, y_scores, target_recall=0.95):
    """
    Find the threshold that gives at least the target recall.
    
    Args:
        y_true: Array-like of true binary labels
        y_scores: Array-like of predicted scores or probabilities
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        tuple: (threshold, achieved_recall)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Add a sentinel for 100% recall case
    if len(thresholds) > 0:
        thresholds = np.append(thresholds, 0)
        precision = np.append(precision[:-1], precision[-1])  # Duplicate last precision for sentinel
        recall = np.append(recall[:-1], 1.0)  # Set 100% recall for sentinel
    
    # Find indices where recall is at least the target
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) > 0:
        # Among thresholds that achieve target recall, find the one with highest precision
        best_idx = valid_indices[np.argmax(precision[valid_indices])]
        threshold = thresholds[best_idx]
        achieved_recall = recall[best_idx]
    else:
        # If no threshold achieves target recall, use lowest threshold (highest recall)
        threshold = thresholds[-1]
        achieved_recall = recall[-1]
    
    return threshold, achieved_recall

def evaluate(y_true, y_pred, y_scores, output_dir, model_name, target_recall=0.95):
    """
    Comprehensive evaluation of a binary classification model.
    
    Args:
        y_true: Array-like of true binary labels
        y_pred: Array-like of predicted binary labels
        y_scores: Array-like of predicted scores or probabilities
        output_dir: Directory to save evaluation outputs
        model_name: Name of the model (used for file naming)
        target_recall: Target recall threshold (default: 0.95)
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Evaluating model '{model_name}' (target recall: {target_recall})")
    
    # Calculate standard metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(y_true, y_scores)
    
    # Calculate WSS@95
    wss = calculate_wss_at_recall(y_true, y_scores, target_recall)
    
    # Find threshold for target recall
    threshold, achieved_recall = find_threshold_for_recall(y_true, y_scores, target_recall)
    
    # Predictions at target recall threshold
    high_recall_preds = (y_scores >= threshold).astype(int)
    high_recall_precision = precision_score(y_true, high_recall_preds)
    high_recall_f1 = f1_score(y_true, high_recall_preds)
    
    # Gather all metrics
    metrics = {
        "model_name": model_name,
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(roc_auc),
        "wss@95": float(wss),
        "high_recall_threshold": float(threshold),
        "high_recall_precision": float(high_recall_precision),
        "high_recall_f1": float(high_recall_f1),
        "high_recall_achieved": float(achieved_recall),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Log key metrics
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"F2 Score: {f2:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"WSS@{target_recall*100:.0f}: {wss:.4f}")
    logger.info(f"High recall ({achieved_recall:.4f}) precision: {high_recall_precision:.4f}")
    
    # Save detailed classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    with open(f"{output_dir}/{model_name}_classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Save metrics as JSON
    with open(f"{output_dir}/{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save ROC curve to plots directory
    plot_roc_curve(y_true, y_scores, f"{plots_dir}/{model_name}_roc_curve.png")
    
    # Plot and save precision-recall curve to plots directory
    plot_precision_recall_curve(y_true, y_scores, f"{plots_dir}/{model_name}_pr_curve.png")
    
    return metrics

def plot_roc_curve(y_true, y_scores, output_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.savefig(output_path)
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_scores, output_path):
    """Plot and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Calculate no-skill line (random classifier)
    no_skill = sum(y_true) / len(y_true)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'No Skill ({no_skill:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()
    
    return pr_auc

def compare_models(model_results_list, output_dir, filename="model_comparison"):
    """
    Create a bar chart comparing multiple models across key metrics.
    
    Args:
        model_results_list: List of (name, metrics_dict) tuples for each model
        output_dir: Directory to save the comparison chart
        filename: Filename for the output chart (default: "model_comparison")
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select important metrics to compare
    metrics = ['roc_auc', 'wss@95', 'f1', 'f2', 'precision', 'recall']
    labels = ['AUC', 'WSS@95', 'F1', 'F2', 'Precision', 'Recall']
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(model_results_list)
    
    # Plot each model's metrics
    for i, (name, results) in enumerate(model_results_list):
        values = [results.get(m, 0) for m in metrics]
        plt.bar(x + (i - len(model_results_list)/2 + 0.5) * width, values, width, label=name)
    
    # Add labels and legend
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, labels)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save the chart
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()
    
    # Also save as CSV for easier analysis
    comparison_data = {
        'Metric': labels
    }
    
    for name, results in model_results_list:
        comparison_data[name] = [results.get(m, 0) for m in metrics]
    
    pd.DataFrame(comparison_data).to_csv(f"{output_dir}/{filename}.csv", index=False)
    
    logger.info(f"Model comparison saved to {output_dir}/{filename}.png and {filename}.csv")