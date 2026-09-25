# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** ML-HACK  
**Team Members:** JAI SURYA  
**Submission Date:** 2026-09-25

---

## 1. Executive Summary
This solution uses a two-stage entity-resolution pipeline. Character 3-gram TF-IDF blocking reduces the search space, then a LightGBM classifier combines fuzzy name, address, and country features to make precision-oriented matches.

---

## 2. Methodology

### 2.1 Problem Analysis
The data contains spelling errors, punctuation changes, legal suffix variations, word-order changes, abbreviated addresses, partial addresses, and missing values. Country is treated as a relational feature rather than a fixed category, so unseen test labels such as France remain supported.

### 2.2 Solution Strategy
Source 1 records are compared with the union of Source 2 and Source 3. Text is lowercased, punctuation is removed, common abbreviations are expanded, and normalized name plus address text is used for blocking and pairwise comparison.

**Approach Type:** Blocking + classifier  
**Core Innovation:** Sparse character n-gram retrieval provides scalable candidate generation while preserving typo-tolerant matching; the final classifier is conservative because the competition metric weights precision more heavily.

---

## 3. Candidate Generation (Blocking)
The combined normalized business name and address are vectorized with character 3-gram TF-IDF. For each Source 1 record, sparse top-k cosine products retain up to 25 candidates above a 0.15 similarity threshold during full inference. The candidate file is written directly from the same pairs scored by the classifier.

- **Blocking keys used:** Character 3-gram TF-IDF over normalized name and address
- **Candidate pairs generated:** Up to 25 per Source 1 record during full inference
- **How you ensured true matches were not lost:** Character n-grams tolerate typos and partial overlaps; the candidate threshold is deliberately broad and top-k is applied only after sparse similarity retrieval.

---

## 4. Matching Model

**Features used:**
- Name features: RapidFuzz ratio, token-sort ratio, and Jaro-Winkler similarity
- Address features: RapidFuzz ratio and token-sort ratio
- Other: Exact normalized country equality, when both country values are present

**Model type:** LightGBM binary classifier  
**Threshold selection method:** Grouped 80/20 holdout by Source 1 entity, selecting the threshold with the highest macro F_0.5. The selected threshold was 0.95.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.95562 on the sample holdout
- **Common false positives (wrong merges):** Businesses with similar names and short or generic addresses can remain ambiguous; the high threshold limits these merges.
- **Common false negatives (missed matches):** Records with heavily abbreviated or landmark-only addresses, and matches whose names share few character n-grams, may fall outside the top-k candidates or below the classifier threshold.

---

## 6. Conclusion
The pipeline is reproducible from the supplied data and produces both required TSV outputs. The main practical lesson is that candidate recall and strict precision control must be tuned together for macro F_0.5, especially because singleton Source 1 entities are part of the score.

---

## Appendix

### A. Code Artefacts
The complete runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). The entry points are `01_create_sample.py`,
`02_blocking.py`, `03_features.py`, `04_model.py`, and
`05_full_pipeline.py`; the README contains the exact commands to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
The sample training run produced 299,500 candidate pairs and a 0.95562 grouped holdout macro F_0.5 at threshold 0.95. Full-test inference streams Source 1 in chunks and writes one row for every test Source 1 entity.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
