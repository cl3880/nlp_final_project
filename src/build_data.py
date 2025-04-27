#!/usr/bin/env python3
"""
Extract PubMed data, dedupe, label relevance, optionally run manual title cleanup, generate detailed report, and save a clean CSV for modeling.
"""
import re
import csv
import argparse
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict


def prompt_fix_titles(
        raw_titles: list[str],
        pubmed_titles: list[str]) -> list[str]:
    """Prompt for manual review of spacing and potential title matches."""
    fixed = []
    for i, title in enumerate(raw_titles, start=1):
        suggestion = re.sub(r'([:;,.!?])(?=[A-Za-z])', r'\1 ', title)
        norm = normalize_title(suggestion)
        best_pm, best_score = max(
            ((pt, SequenceMatcher(None, norm, normalize_title(pt)).ratio())
             for pt in pubmed_titles),
            key=lambda x: x[1]
        )

        if suggestion == title and best_score == 1.0:
            fixed.append(title)
            continue

        print(f"\n[{i}/{len(raw_titles)}]")
        print("Original      :", title)
        if suggestion != title:
            print("Auto-spacing  :", suggestion)
        if best_score >= 0.85:
            print(f"PubMed match  : {best_pm}  (score={best_score:.2f})")

        choice = input(
            "  [a] Auto-spacing  [p] Use PubMed title  [e] Edit  [k] Keep original: ").strip().lower()
        if choice == 'p' and best_score >= 0.85:
            fixed_title = best_pm
        elif choice == 'a' and suggestion != title:
            fixed_title = suggestion
        elif choice == 'e':
            fixed_title = input("Enter corrected title: ").strip()
        else:
            fixed_title = title

        fixed.append(fixed_title)
    return fixed


def get_relevant_titles(
        relevant_file: str,
        pubmed_titles: list[str]) -> list[str]:
    """Load titles to review."""
    text = Path(relevant_file).read_text(encoding='utf-8')
    raw = [t.strip() for t in text.split('\n\n') if t.strip()]
    print(f"\nLoaded {len(raw)} titles from {relevant_file}. Cleaning...")
    cleaned = prompt_fix_titles(raw, pubmed_titles)
    return cleaned


def clean_text(s: str) -> str:
    s = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", s)
    return s.replace("\u00a0", " ").strip()


def extract_pubmed(pubmed_file: str) -> list[dict]:
    records = []
    current = {}
    lines = Path(pubmed_file).read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        if raw.startswith('PMID- '):
            if current:
                records.append(current)
            current = {
                'pmid': raw.split('PMID- ')[1].strip(),
                'mesh_terms': [],
                'keywords': [],
                'publication_types': [],
                'dois': []
            }
        elif raw.startswith('TI  - '):
            title = raw.split('TI  - ')[1].strip()
            j = i + 1
            while j < len(lines) and lines[j].startswith('      '):
                title += ' ' + lines[j].strip()
                j += 1
            current['title'] = clean_text(title)
            i = j - 1
        elif raw.startswith('AB  - '):
            abstract = raw.split('AB  - ')[1].strip()
            j = i + 1
            while j < len(lines) and lines[j].startswith('      '):
                abstract += ' ' + lines[j].strip()
                j += 1
            current['abstract'] = clean_text(abstract)
            i = j - 1
        elif raw.startswith('DP  - '):
            date_str = raw.split('DP  - ')[1].strip()
            current['publication_date'] = date_str
            year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
            if year_match:
                current['publication_year'] = int(year_match.group(0))
        elif raw.startswith('JT  - '):
            current['journal'] = raw.split('JT  - ')[1].strip()
        elif raw.startswith('TA  - ') and not current.get('journal'):
            current['journal'] = raw.split('TA  - ')[1].strip()
        elif raw.startswith('LID - '):
            doi = raw.split('LID - ')[1].split()[0]
            current['dois'].append(doi)
        elif raw.startswith('AID - '):
            aid = raw.split('AID - ')[1].split()[0]
            current['dois'].append(aid)
        elif raw.startswith('MH  - '):
            current['mesh_terms'].append(raw.split('MH  - ')[1].strip())
        elif raw.startswith('OT  - '):
            current['keywords'].append(raw.split('OT  - ')[1].strip())
        elif raw.startswith('PT  - '):
            current['publication_types'].append(raw.split('PT  - ')[1].strip())
        elif raw.startswith('VI  - '):
            current['volume'] = raw.split('VI  - ')[1].strip()
        elif raw.startswith('IP  - '):
            current['issue'] = raw.split('IP  - ')[1].strip()
        elif raw.startswith('PG  - '):
            current['pages'] = raw.split('PG  - ')[1].strip()
        elif raw.startswith('LA  - '):
            current['language'] = raw.split('LA  - ')[1].strip()
        i += 1
    if current:
        records.append(current)
    print(f"Extracted {len(records)} records from {pubmed_file}")
    return records


def normalize_title(title: str) -> str:
    t = title
    t = re.sub(r"-", " ", t)
    t = re.sub(r"[^\w\s\[\]]", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def dedupe(records: list[dict]) -> tuple[list[dict], list[dict]]:
    seen = set()
    unique, dupes = [], []
    for r in records:
        key = r['dois'][0] if r.get(
            'dois') else normalize_title(r.get('title', ''))
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
        else:
            dupes.append(r)
    print(f"Deduped: kept {len(unique)}, dropped {len(dupes)} duplicates")
    return unique, dupes


def create_detailed_report(
    report_file: str,
    dupes: list[dict],
    unmatched_refs: list[str],
    records: list[dict]
) -> None:
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as f:
        f.write("=== DUPLICATES DROPPED ===\n")
        for r in dupes:
            f.write(f"- PMID {r['pmid']}: {r.get('title','')}\n")
        f.write(f"Total duplicates dropped: {len(dupes)}\n\n")

        f.write("=== UNMATCHED REFERENCE TITLES (no similarity) ===\n")
        for i, title in enumerate(unmatched_refs, 1):
            norm = normalize_title(title)
            f.write(f"{i}. {title}\n   Normalized: {norm}\n\n")

        f.write("=== TITLES MATCHING MULTIPLE ARTICLES ===\n")
        f.write("Report complete.\n")
    print(f"Detailed report written to {report_file}")


def save_csv(records: list[dict], output_csv: str) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = [
        'pmid',
        'title',
        'abstract',
        'publication_date',
        'publication_year',
        'journal',
        'volume',
        'issue',
        'pages',
        'language',
        'relevant']
    lists = ['dois', 'publication_types', 'mesh_terms', 'keywords']
    fields = base + lists
    with output_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = {c: r.get(c, '') for c in base}
            for l in lists:
                row[l] = ";".join(r.get(l, []))
            w.writerow(row)
    print(f"Saved {len(records)} records to {output_csv}")


def mark_relevant_articles(records: list[dict], relevant_titles: list[str]) -> tuple[list[dict], list[str]]:
    norm_ref = { normalize_title(t): t for t in relevant_titles }
    matched = set()

    for r in records:
        title = r.get('title', "")
        norm  = normalize_title(title)
        if norm in norm_ref:
            r['relevant'] = True
            matched.add(norm)
        else:
            r['relevant'] = False

    unmatched = [ norm_ref[n] for n in norm_ref if n not in matched ]

    total_records  = len(records)
    total_relevant = sum(1 for r in records if r.get('relevant', False))

    print(f"Original relevant titles:            {len(relevant_titles)}")
    print(f"Unique normalized relevant titles:    {len(norm_ref)}")
    print(f"Records marked as relevant:           {total_relevant} / {total_records}")
    print(f"No-match titles (for manual review): {len(unmatched)}")

    return records, sorted(unmatched)

def get_deduplication_key(record):
    if record.get('dois'):
        doi  = record['dois'][0]
        pmid = record['pmid']
        return f"doi:{doi}|pmid:{pmid}"

    return f"title:{normalize_title(record.get('title',''))}"

def manual_review_duplicates(groups):
    """Review groups with multiple entries and decide how to handle them."""
    reviewed_groups = {}
    
    for key, entries in sorted(
            groups.items(), 
            key=lambda x: len(x[1]), 
            reverse=True):
        if len(entries) <= 1:
            reviewed_groups[key] = entries
            continue
            
        print(f"\nFound {len(entries)} entries with key: {key}")
        for i, entry in enumerate(entries):
            print(f"  {i+1}. {entry.get('title', 'No title')} (PMID: {entry.get('pmid', 'Unknown')})")
        
        choice = input("Enter entry numbers to keep (comma-separated), or 'a' to keep all, 'm' to merge: ").strip().lower()
        
        if choice == 'a':
            # Keep all as separate entries
            reviewed_groups[key] = entries
        elif choice == 'm':
            # Merge entries
            base = entries[0].copy()
            base['pmids'] = [r['pmid'] for r in entries]
            base['all_abstracts'] = [r.get('abstract', '') for r in entries]
            base['all_mesh_terms'] = sum((r.get('mesh_terms', []) for r in entries), [])
            base['all_keywords'] = sum((r.get('keywords', []) for r in entries), [])
            reviewed_groups[key] = [base]
        else:
            try:
                # Keep only selected entries
                indices = [int(i.strip()) - 1 for i in choice.split(',') if i.strip()]
                valid_indices = [i for i in indices if 0 <= i < len(entries)]
                if valid_indices:
                    reviewed_groups[key] = [entries[i] for i in valid_indices]
                else:
                    print("  No valid selection, keeping all.")
                    reviewed_groups[key] = entries
            except ValueError:
                print("  Invalid input, keeping all.")
                reviewed_groups[key] = entries
                
    # Flatten the dictionary of lists into a single list
    result = []
    for entries in reviewed_groups.values():
        result.extend(entries)
    return result

def main():
    parser = argparse.ArgumentParser(
        description="Build clean PubMed dataset: positional args for pubmed_txt, relevant_txt, output_csv, report_file, use -m for manual review.")
    parser.add_argument('pubmed_txt', help='Path to raw PubMed .txt file')
    parser.add_argument(
        'relevant_txt',
        help='Path to text file of relevant titles')
    parser.add_argument(
        'output_csv',
        help='Output CSV filename (will be placed in /data/processed/ if no directory)')
    parser.add_argument(
        'report_file',
        help='Report filename (will be placed in reports/ if no directory)')
    parser.add_argument(
        '-m',
        '--manual',
        action='store_true',
        help='Run manual review')
    args = parser.parse_args()

    out_path = Path(args.output_csv)
    if out_path.parent in (Path('.'), Path()):
        out_path = Path('data/processed') / out_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_path = Path(args.report_file)
    if report_path.parent in (Path('.'), Path()):
        report_path = Path('reports') / report_path.name
    report_path.parent.mkdir(parents=True, exist_ok=True)

    records = extract_pubmed(args.pubmed_txt)
    pubmed_titles = [r['title'] for r in records]

    # Group by more robust deduplication key
    groups = defaultdict(list)
    for r in records:
        key = get_deduplication_key(r)
        groups[key].append(r)
    
    # Show statistics about grouping
    print(f"Grouped {len(records)} records into {len(groups)} unique keys")
    multi_entries = sum(1 for entries in groups.values() if len(entries) > 1)
    print(f"Found {multi_entries} keys with multiple entries")
    
    # Process groups based on manual review or automatic merging
    if args.manual:
        merged = manual_review_duplicates(groups)
        all_dupes = []  # We don't track dupes the same way with manual review
    else:
        # Automatic merging
        merged, all_dupes = [], []
        for grp in groups.values():
            if len(grp) == 1:
                merged.append(grp[0])
            else:
                base = grp[0].copy()
                base['pmids'] = [r['pmid'] for r in grp]
                base['all_abstracts'] = [r.get('abstract', '') for r in grp]
                base['all_mesh_terms'] = sum(
                    (r.get('mesh_terms', []) for r in grp), [])
                base['all_keywords'] = sum(
                    (r.get('keywords', []) for r in grp), [])
                merged.append(base)
                all_dupes.extend(grp[1:])  # First one is kept, rest are dupes

    print(f"After deduplication: {len(merged)} records")

    if args.manual:
        refs = get_relevant_titles(args.relevant_txt, pubmed_titles)
    else:
        refs = [t.strip()
            for t in Path(args.relevant_txt)
                        .read_text(encoding='utf-8')
                        .split('\n\n')
            if t.strip()]

    labeled, unmatched = mark_relevant_articles(merged, refs)
    print(
        f"Labeled relevant: {sum(r.get('relevant', False) for r in labeled)} / {len(labeled)}")
    print(f"Duplicates detected: {len(all_dupes)}")
    print(f"No-match titles: {len(unmatched)}")

    create_detailed_report(str(report_path), all_dupes, unmatched, labeled)
    save_csv(labeled, str(out_path))


if __name__ == '__main__':
    main()
