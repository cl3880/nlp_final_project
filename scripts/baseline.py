# models/baseline.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity

import logging

logger = logging.getLogger(__name__)

class CosineSimilarityClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple 'classifier' that fits TF-IDF, computes centroid of positive docs,
    then scores new docs by cosine similarity to that centroid.
    """
    def __init__(self, threshold=None):
        # threshold: if None, you'll pick it later to hit target recall
        self.threshold = threshold

    def fit(self, X, y):
        # X: array of strings; y: boolean mask
        tfidf = X  # expects X already vectorized if used after TfidfVectorizer
        positives = tfidf[y.astype(bool)]
        
        # compute centroid and ensure it's in the right format
        # Convert to dense if it's a sparse matrix
        if hasattr(positives, 'toarray'):
            # Handle sparse matrix
            self.centroid_ = positives.toarray().mean(axis=0).reshape(1, -1)
        else:
            # Handle regular array
            self.centroid_ = positives.mean(axis=0).reshape(1, -1)
        return self

    def predict_proba(self, X):
        # X: TF-IDF vectors
        # Convert input to appropriate format for cosine_similarity
        if hasattr(X, 'toarray'):
            # Handle sparse matrix
            X_array = X.toarray()
        else:
            # Handle regular array
            X_array = X
            
        # Calculate cosine similarity
        sims = cosine_similarity(X_array, self.centroid_).ravel()
        
        # Two-column: prob for class 0 = 1-s, class 1 = s
        return np.vstack([1 - sims, sims]).T

    def predict(self, X):
        # Get similarity scores
        probs = self.predict_proba(X)[:, 1]
        
        # Apply threshold
        if self.threshold is None:
            # fallback to 0.5
            return (probs >= 0.5).astype(int)
        return (probs >= self.threshold).astype(int)


def make_tfidf_cosine_pipeline(max_features=10000,
                               ngram_range=(1, 2),
                               min_df=3,
                               text_columns=['title','abstract']):
    """
    Combines TextCombiner, TfidfVectorizer, and CosineSimilarityClassifier.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    return Pipeline([
        ('combiner', TextCombiner(text_columns)),
        ('tfidf', TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english',
            sublinear_tf=True)),
        ('cosine', CosineSimilarityClassifier())
    ])

class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer to combine multiple text columns into a single text field.
    Useful when processing DataFrames with title and abstract columns.
    """
    def __init__(self, text_columns=['title', 'abstract']):
        self.text_columns = text_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Combine text columns with a space between them."""
        if hasattr(X, 'iloc'):
            logger.debug(f"Combining text columns: {self.text_columns}")
            combined = X[self.text_columns[0]].fillna('')
            for col in self.text_columns[1:]:
                if col in X.columns:
                    combined = combined + ' ' + X[col].fillna('')
            return combined.values
        return X

def make_tfidf_logreg_pipeline(max_features=10000, 
                              ngram_range=(1, 2),
                              min_df=3,
                              text_columns=['title', 'abstract'],
                              C=1.0,
                              class_weight='balanced'):
    """
    Create a scikit-learn pipeline combining TF-IDF and logistic regression.
    
    Args:
        max_features: Maximum number of features for TF-IDF (default: 10000)
        ngram_range: Range of n-grams to include (default: (1, 2))
        min_df: Minimum document frequency for terms (default: 3)
        text_columns: List of text columns to combine (default: ['title', 'abstract'])
        C: Inverse regularization strength (default: 1.0)
        class_weight: Class weights for imbalanced data (default: 'balanced')
        
    Returns:
        A scikit-learn Pipeline object
    """
    logger.info("Creating TF-IDF + LogReg pipeline")
    logger.debug(f"Parameters: max_features={max_features}, ngram_range={ngram_range}, "
                f"min_df={min_df}, C={C}, class_weight={class_weight}")
    
    return Pipeline([
        ('combiner', TextCombiner(text_columns)),
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

def baseline_param_grid():
    """
    Define parameter grid for baseline model hyperparameter search.
    
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    return {
        'tfidf__max_features': [5000, 10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__min_df': [2, 3, 5],
        'clf__C': [0.01, 0.1, 1, 10, 100],
        'clf__class_weight': ['balanced'],
        'clf__penalty': ['l1', 'l2'],
        'clf__solver': ['liblinear'],
    }

def get_top_features(model, n=20, class_names=['Irrelevant', 'Relevant']):
    """
    Extract the most important features for each class from a trained model.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and LogisticRegression
        n: Number of top features to extract (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        DataFrame containing feature names and their coefficients
    """
    import pandas as pd
    
    try:
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        features_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })
        
        features_df['abs_coef'] = features_df['coefficient'].abs()
        features_df = features_df.sort_values('abs_coef', ascending=False)
        
        features_df['class'] = features_df['coefficient'].apply(
            lambda x: class_names[1] if x > 0 else class_names[0]
        )
        
        top_relevant = features_df[features_df['coefficient'] > 0].sort_values(
            'coefficient', ascending=False
        ).head(n)
        
        top_irrelevant = features_df[features_df['coefficient'] < 0].sort_values(
            'coefficient', ascending=True
        ).head(n)
        
        return pd.concat([top_relevant, top_irrelevant])
        
    except Exception as e:
        logger.error(f"Error extracting top features: {e}")
        return pd.DataFrame()

def plot_top_features(model, output_path, n=20, class_names=['Irrelevant', 'Relevant']):
    """
    Plot and save the most important features for each class.
    
    Args:
        model: Trained pipeline with TfidfVectorizer and LogisticRegression
        output_path: Path to save the plot
        n: Number of top features to show (default: 20)
        class_names: Names of the classes (default: ['Irrelevant', 'Relevant'])
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    features_df = get_top_features(model, n, class_names)
    
    if features_df.empty:
        logger.warning("No features to plot")
        return
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    relevant_features = features_df[features_df['class'] == class_names[1]].head(n)
    sns.barplot(x='coefficient', y='feature', data=relevant_features)
    plt.title(f'Top {n} Features Indicating {class_names[1]}')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    irrelevant_features = features_df[features_df['class'] == class_names[0]].head(n)
    sns.barplot(x='coefficient', y='feature', data=irrelevant_features)
    plt.title(f'Top {n} Features Indicating {class_names[0]}')
    plt.tight_layout()
    
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Top features plot saved to {output_path}")