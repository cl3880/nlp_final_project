# config.py
"""
Configuration settings for the systematic review classification project.
"""
PATHS = {
    "data_raw":        "data/raw",
    "data_processed":  "data/processed",
    "results_dir":     "results",
    "logs_dir":        "results/logs",
    "baseline_dir":    "results/baseline",
    "build_data_dir":  "results/build_data"
}

MODEL_CONFIG = {
    "paths": {
        "output_dir":  "results/models",
        "cache_dir":   "cache",
        "log_file":    "results/logs/training.log",
        "models_dir":  "results/baseline/models",
        "metrics_dir": "results/baseline/metrics",
        "plots_dir":   "results/baseline/plots",
        "analysis_dir":"results/baseline/analysis"
    },
    
    "tfidf": {
        "max_features": 10000,
        "ngram_range": (1, 2),
        "min_df": 3,
        "sublinear_tf": True,
        "stop_words": "english"
    },
    
    "classifier": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "random_state": 42
    },
    
    "evaluation": {
        "recall_threshold": 0.95,
        "cv_folds": 5
    },
    
    "regex": {
        "sample_size": r"(?:n\s*=\s*|sample\s+size\s*(?:of|was|:|=)\s*)(\d+)",
        "occlusion": r"(?:occlusion|obliteration)\s+(?:rate|percentage|ratio)",
        "adult_age": r"(?:\b(?:adult|mature|grown)\b|(?:older|elderly)\s+(?:than|adults?)|\b(?:age[ds]?|older)\s+(?:\d+|\w+teen))",
        "pediatric": r"\b(?:child|children|pediatric|infant|adolescent|neonatal|juvenile)\b",
        "age_range": r"(?:age|aged)\s+(?:range|between|from)?\s*(?:was|:)?\s*(\d+)(?:\s*[-–]\s*|\s+to\s+)(\d+)",
        "radiosurgery": r"\b(?:radiosurg|gamma\s+knife|cyberknife|novalis|linear\s+accelerator\s+(?:based\s+)?radiosurg)\w*\b",
        "brain_avm": r"\b(?:brain|cerebral|intracranial)\s+(?:arteriovenous\s+malformation|avm)\b"
    },
    
    "criteria": {
        "min_year": 2000,
        "sample_size_min": 10,
        "adult_age_min": 18
    }
}