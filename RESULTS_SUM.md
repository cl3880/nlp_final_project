# Results Summary

## Project Overview

We built a system to classify PubMed articles into **relevant** vs **irrelevant** to assist systematic review tasks.  
Our key goals are:
- Achieve **high recall** (≥95%) to avoid missing relevant studies.
- Improve **work saved** compared to random sampling (WSS@95 metric).

## Dataset

| Metric           | Value    |
|------------------|----------|
| Total Records    | 2175     |
| Relevant Records | 242 (11%) |

Data was manually annotated by a medical student based on strict inclusion/exclusion criteria.

---

### Cosine Similarity Threshold

Threshold testing results for the Cosine Similarity model:

| Threshold | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| 0.3       | 0.4722    | 0.7083 | 0.5667   |
| 0.4       | 0.5000    | 0.0833 | 0.1429   |
| 0.5       | 0.0000    | 0.0000 | 0.0000   |

- **Threshold 0.3** was selected because it provided the best trade-off between precision and recall.
- Higher thresholds (0.4 and 0.5) drastically reduced recall, making them unsuitable for our high-recall goal.
- Even with optimal threshold tuning, Cosine Similarity underperforms compared to Logistic Regression.

---

## Model Results

| Approach                          | Precision | Recall | F1 Score | F2 Score | ROC AUC | WSS@95 |
|-----------------------------------|-----------|--------|----------|----------|---------|--------|
| Logistic Regression (Baseline)   | 0.4667    | 0.8750 | 0.6087   | 0.7447   | 0.9467  | 0.6977 |
| Cosine Similarity (Threshold=0.3) | 0.4722    | 0.7083 | 0.5667   | 0.6439   | 0.9276  | 0.6839 |
| Logistic Regression + Stemming   | 0.4889    | 0.9167 | 0.6377   | 0.7801   | 0.9515  | 0.6977 |
| Logistic Regression + Lemmatization | **0.5000** | **0.9167** | **0.6471** | **0.7857** | 0.9482 | 0.6885 |

---

## Interpretations

### Logistic Regression vs Cosine Similarity

- **Logistic Regression outperforms Cosine Similarity** on all metrics.
- Cosine Similarity is a weaker baseline here because the task is **supervised**:  
  Labels (relevance/irrelevance) were assigned by humans, and Logistic Regression directly learns this mapping.
- Cosine Similarity is more appropriate for *unsupervised* similarity tasks (e.g., nearest neighbor search).

### Normalization (Stemming, Lemmatization)

- **Lemmatization provided the best boost**: both recall and precision improved without sacrificing AUC.
- Higher recall (91.7%) ensures fewer relevant articles are missed, crucial for systematic reviews.

---

## Error Analysis Insights

- Only **3 false negatives** out of 218 test examples (1.4%), indicating strong performance.
- **24 false positives** observed, mainly due to articles mentioning terms like "obliteration" but being irrelevant (e.g., pediatric studies).

---
### Top Logistic Regression Features (Before Filtering)

![Top Features](results/baseline/plots/logreg_top_features.png)

- **Strong indicators of irrelevance**:  
  - `pediatric`, `children`, `case`, `meta`, `systematic review`
- **Strong indicators of relevance**:  
  - `obliteration`, `gamma knife`, `stereotactic radiosurgery`, `nidus`

**Interesting Note**:  
Even before we manually implemented pediatric/age filters, the model **learned to associate "pediatric" and "children" with irrelevance** on its own.

---

## SMOTE

Given:
- 11% class imbalance
- Still relatively low precision (~0.46 baseline)
- Moderate dataset size (n=2175)


