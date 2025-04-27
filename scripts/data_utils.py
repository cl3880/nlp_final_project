# utils/data_utils.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_data(filepath):
    """
    Load dataset from CSV file, ensuring proper data types.
    
    Args:
        filepath: Path to the CSV file containing the dataset
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Convert relevant column to boolean if it exists
        if 'relevant' in df.columns:
            df['relevant'] = df['relevant'].astype(bool)
        
        # Fill missing abstracts with empty string
        if 'abstract' in df.columns:
            df['abstract'] = df['abstract'].fillna('')
        
        # Convert publication_year to integer if possible
        if 'publication_year' in df.columns:
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
            df['publication_year'] = df['publication_year'].fillna(0).astype(int)
        
        logger.info(f"Successfully loaded {len(df)} records")
        
        # Report class distribution if relevant column exists
        if 'relevant' in df.columns:
            positive = df['relevant'].sum()
            logger.info(f"Class distribution: {positive} relevant, {len(df) - positive} irrelevant")
            
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def make_splits(df, test_size=0.1, val_size=0.1, stratify=True, seed=42):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: DataFrame containing the dataset
        test_size: Proportion of data to use for testing (default: 0.1)
        val_size: Proportion of remaining data to use for validation (default: 0.1)
        stratify: Whether to stratify splits by the 'relevant' column (default: True)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        train, val, test: Three DataFrames with the respective splits
    """
    logger.info("Splitting data into train, validation, and test sets")
    
    # Determine stratification
    stratify_col = df['relevant'] if stratify and 'relevant' in df.columns else None
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=seed,
        stratify=stratify_col
    )
    
    # Update stratification for second split
    if stratify_col is not None:
        stratify_col = train_val['relevant']
    
    # Second split: separate train and validation sets
    train, val = train_test_split(
        train_val, 
        test_size=val_size/(1-test_size),
        random_state=seed,
        stratify=stratify_col
    )
    
    logger.info(f"Split sizes: {len(train)} train, {len(val)} validation, {len(test)} test")
    
    # Report class distribution in each split if relevant column exists
    if 'relevant' in df.columns:
        logger.info(f"Train positive: {train['relevant'].sum()} ({train['relevant'].mean()*100:.1f}%)")
        logger.info(f"Validation positive: {val['relevant'].sum()} ({val['relevant'].mean()*100:.1f}%)")
        logger.info(f"Test positive: {test['relevant'].sum()} ({test['relevant'].mean()*100:.1f}%)")
    
    return train, val, test

def combine_text_fields(df, columns=['title', 'abstract']):
    """
    Combine multiple text columns into a single text field.
    
    Args:
        df: DataFrame containing the dataset
        columns: List of column names to combine
        
    Returns:
        Series with combined text
    """
    combined = df[columns[0]].fillna('')
    
    for col in columns[1:]:
        combined = combined + ' ' + df[col].fillna('')
    
    return combined