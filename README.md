# NLP Final Project: Systematic Review Automation

This repository automates the extraction, deduplication, and relevance classification of PubMed articles to support medical systematic reviews.

---

## Project Structure

```
nlp_final_project/
├── data/
│   ├── raw/         # Raw PubMed export and relevant title lists
│   └── processed/   # Cleaned, finalized CSVs
├── results/
│   ├── baseline/    # Baseline model outputs
│   └── baseline_with_normalization/ # Normalized model outputs
├── reports/         # Logs, reports, manual cleanup notes
├── scripts/         # Scripts for data preparation, training, evaluation
└── README.md        # This file
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Workflow

### 1. (Optional) Data Preparation
*You only need to do this if starting from scratch. `data_final_cleaned.csv` already exists.*

```bash
python scripts/01_build_data.py data/raw/pubmed_results.txt data/raw/relevant_titles.txt data/processed/data_final.csv reports/report.txt -m
python scripts/02_prepare_data.py --input data/processed/data_final.csv --output data/processed/data_final_cleaned.csv
```

---

### 2. Train Baseline Model (TF-IDF + Logistic Regression + Cosine Similarity)

```bash
python scripts/03_train_baseline.py --data data/processed/data_final_cleaned.csv --output-dir results/baseline
```

Optional: Specify a cosine similarity threshold:

```bash
python scripts/03_train_baseline.py --data data/processed/data_final_cleaned.csv --output-dir results/baseline --cos-thresh 0.3
```

---

### 3. Error Analysis

```bash
python scripts/analyze_logreg_predictions.py
```

Outputs will save to `results/baseline/error_analysis/`.

---

### 4. Compare Normalization Techniques (Stemming, Lemmatization)

```bash
python scripts/compare_normalizations.py --data data/processed/data_final_cleaned.csv --output-dir results/baseline_with_normalization
```

---

## Medical Student's Systematic Review Screening Criteria

| Category                         | Include                                                                          | Exclude                                                                                                                                                   |
|----------------------------------|----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Population**                   | Patients ≥18 years old with brain AVMs                                           | Only pediatric populations, pregnant patients, dural/pial arteriovenous fistulas, vein of Galen malformations, cavernous malformations                    |
| **Age Mention**                  | No mention of patients under 18                                                  | Any mention of patients under 18 (e.g., "age range 4–98")                                                                                                 |
| **Automatic Exclusion Keywords** | —                                                                                | hypofractionated, proton beam therapy, fractionated stereotactic radiotherapy/surgery, tomotherapy                                                        |
| **Minimum Sample Size**          | If no patient number mentioned, **keep**                                         | If explicitly <30 patients, **exclude**                                                                                                                   |
| **Publication Year**             | Published ≥2000                                                                  | Published before 2000                                                                                                                                     |
| **Language**                     | English or French                                                               | Non-English, non-French                                                                                                                                   |
| **Outcome Reporting**            | Mentions methodology for AVM obliteration rate                                  | No obliteration rate mentioned                                                                                                                            |
| **Study Type**                   | Clinical trials, cohort studies, case series, systematic reviews                | Meta-analyses, literature reviews, case reports <10 patients                                                                                              |
| **Treatment Method**             | Radiosurgery methods: Gamma Knife, CyberKnife, Novalis, LINAC-based techniques   | Non-radiosurgical treatments                                                                                                                              |

---

## Outputs

- **data/processed/data_final_cleaned.csv**: Final modeling dataset
- **results/baseline/**: Baseline models and evaluation plots
- **results/baseline_with_normalization/**: Results after applying normalization
- **reports/**: Dedupe reports, manual edits

---

## Notes

- Logistic Regression model consistently outperforms Cosine Similarity.
- Text normalization (especially lemmatization) improves precision and recall.
- Error analysis shows that "pediatric" and "children" were top predictors of irrelevant studies **before** implementing stricter age-based filtering.

---
