# run_baseline.py
#!/usr/bin/env python3
"""
End-to-end pipeline for baseline model training and evaluation.
This script handles data preparation, model training, and result visualization.
"""

import os
import argparse
import logging
import config
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run end-to-end pipeline for baseline model')
    parser.add_argument('--raw-data', default=os.path.join(config.PATHS["data_raw"], "data_final.csv"), 
                       help='Path to raw data CSV')
    parser.add_argument('--processed-data', default=os.path.join(config.PATHS["data_processed"], "data_final_processed.csv"),
                       help='Path to save processed data')
    parser.add_argument('--output-dir', default=config.PATHS["baseline_dir"],
                       help='Directory for outputs')
    parser.add_argument('--skip-data-prep', action='store_true',
                       help='Skip data preparation step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(config.PATHS["logs_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(args.processed_data), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting baseline pipeline")
    
    # Step 1: Prepare data
    if not args.skip_data_prep:
        logger.info("Preparing data...")
        os.system(f"python scripts/02_prepare_data.py --input {args.raw_data} --output {args.processed_data}")
    else:
        logger.info("Skipping data preparation")
    
    # Step 2: Train and evaluate model
    if not args.skip_training:
        logger.info("Training baseline model...")
        os.system(f"python scripts/03_train_baseline.py --data {args.processed_data} --output-dir {args.output_dir}")
    else:
        logger.info("Skipping model training")
    
    # Step 3: Generate summary of results
    metrics_file = os.path.join(args.output_dir, "metrics", "baseline_test_metrics.json")
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        logger.info("Baseline model performance:")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"WSS@95: {metrics['wss@95']:.4f}")
    
    logger.info("Baseline pipeline complete")

if __name__ == "__main__":
    main()