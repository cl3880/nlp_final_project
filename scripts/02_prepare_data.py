#!/usr/bin/env python3
"""
Prepare and clean data for the systematic review classification project.
This script loads raw data, applies necessary preprocessing, and saves processed datasets.
"""

import os
import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.PATHS["logs_dir"], "data_preparation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean text fields by removing excessive whitespace and special characters."""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    
    return text.strip()

def extract_year(date_str):
    """Extract year from a date string."""
    if pd.isna(date_str) or date_str is None:
        return np.nan
    
    year_match = re.search(r'(19|20)\d{2}', str(date_str))
    if year_match:
        return int(year_match.group(0))
    
    return np.nan

def prepare_data(input_file, output_file):
    """
    Prepare and clean the dataset.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed CSV file
    """
    logger.info(f"Loading data from {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        original_rows = len(df)
        logger.info(f"Loaded {original_rows} records")
        
        text_cols = ['title', 'abstract']
        for col in text_cols:
            if col in df.columns:
                logger.info(f"Cleaning {col} column")
                df[col] = df[col].apply(clean_text)
        
        if 'publication_year' in df.columns:
            logger.info("Processing publication_year column")
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
        elif 'publication_date' in df.columns:
            logger.info("Extracting year from publication_date")
            df['publication_year'] = df['publication_date'].apply(extract_year)
        
        if 'relevant' in df.columns:
            logger.info("Converting relevant column to boolean")
            df['relevant'] = df['relevant'].astype(bool)
        
        if 'title' in df.columns:
            missing_title = df['title'].isna() | (df['title'] == '')
            if missing_title.any():
                logger.warning(f"Dropping {missing_title.sum()} rows with missing titles")
                df = df[~missing_title]
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} processed records to {output_file}")
        logger.info(f"Processing complete: {original_rows - len(df)} records removed")
        
        return df
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Prepare data for systematic review classification')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output processed CSV file')
    args = parser.parse_args()
    
    os.makedirs(config.PATHS["logs_dir"], exist_ok=True)
    
    prepare_data(args.input, args.output)

if __name__ == "__main__":
    main()