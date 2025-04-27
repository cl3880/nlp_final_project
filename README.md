# NLP Final Project: Systematic Review Automation

This repository automates the extraction, deduplication, and relevance labeling of PubMed articles for systematic review tasks.

## Project Structure

```
nlp_final_project/
├── data/
│   ├── raw/
│   │   ├── pubmed_results.txt   # MEDLINE-format PubMed export
│   │   └── relevant_titles.txt  # Reference list of titles to mark as relevant
│   └── processed/
│       └── data_final.csv       # Labeled dataset output from build_data.py
├── reports/
│   └── report.txt               # Detailed dedupe & unmatched report
├── requirements.txt             # Python dependencies
├── src/
│   └── build_data.py            # Script: extract, dedupe, label, report
└── README.md                    # Project overview and instructions
```

## Setup

1. **Create and activate a virtual environment** (Linux/macOS):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   On Windows (PowerShell):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

Run the dataset-building script:

```bash
python src/build_data.py \
  data/raw/pubmed_results.txt \
  data/raw/relevant_titles.txt \
  data/processed/output.csv \
  reports/report.txt [OPTIONS]
```

- **Positional arguments**:
  1. Path to raw PubMed `.txt` file
  2. Path to relevant titles `.txt`
  3. Output CSV path
  4. Report file path

- **Options**:
  - `-m`, `--manual`: Launch interactive title cleanup & duplicate-review prompts.

### Batch mode

```bash
python src/build_data.py \
  data/raw/pubmed_results.txt \
  data/raw/relevant_titles.txt \
  data/processed/output.csv \
  reports/report.txt
```

### Interactive mode

```bash
python src/build_data.py \
  data/raw/pubmed_results.txt \
  data/raw/relevant_titles.txt \
  data/processed/output.csv \
  reports/report.txt \
  -m
```

During `--manual`, you can:
- **Keep / edit** title spacing
- **Keep / merge** deduplication groups

## Outputs

- **`data/processed/output.csv`**: CSV with the following columns:
  - **Core**: `pmid`, `title`, `abstract`, `publication_date`, `publication_year`, `journal`, `volume`, `issue`, `pages`, `language`, `relevant`
  - **List** (semi-colon separated): `dois`, `publication_types`, `mesh_terms`, `keywords`

- **`reports/report.txt`**: Report detailing:
  - **Duplicates** dropped or merged
  - **Unmatched titles** for manual review
  - **Summary stats** (counts of records, matched/unmatched)

## Next Steps

- Integrate `output.csv` into model training pipelines.
- Fine-tune `normalize_title` for additional edge cases.
- Adjust manual review workflow or thresholds as needed.

