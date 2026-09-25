# ?? Business Entity Resolution: The Complete Master Guide & Progress Log
**Project:** End-to-End Entity Resolution for Large-Scale Commercial Platforms
**Target Metric:** Macro-Averaged F_0.5 Score (Precision-Weighted)
**Status:** Phases 1 to 4 Completed (Current Holdout Validation Score: **95.6%**)

---

# ?? Table of Contents
1. [What is Entity Resolution? (Layman Explanation for the Team)](#1-what-is-entity-resolution-layman-explanation)
2. [Why Simple Solutions Fail: The 5 Types of Data Noise](#2-why-simple-solutions-fail-the-5-types-of-data-noise)
3. [The Secret to Winning: Mastering the F_0.5 Metric & Singletons](#3-the-secret-to-winning-mastering-the-f05-metric--singletons)
4. [The 2-Stage Master Architecture](#4-the-2-stage-master-architecture)
5. [Deep Dive into Every Mathematical Algorithm Used](#5-deep-dive-into-every-mathematical-algorithm-used)
   - [A. Character 3-Gram TF-IDF Vectorization](#a-character-3-gram-tf-idf-vectorization)
   - [B. Cosine Similarity & Sparse Matrix Multiplication](#b-cosine-similarity--sparse-matrix-multiplication)
   - [C. The 4 String Distance Algorithms (Levenshtein, Jaro-Winkler, Token Sort, Jaccard)](#c-the-4-string-distance-algorithms)
   - [D. How LightGBM Boosted Trees Work on Tabular Similarities](#d-how-lightgbm-boosted-trees-work)
6. [Phase-by-Phase Development Log (What We Did & How to Run It)](#6-phase-by-phase-development-log)
   - [Phase 1: Environment Setup & Smart Sampling](#phase-1-smart-sandboxing--stratified-sampling)
   - [Phase 2: Candidate Generation (Blocking)](#phase-2-candidate-generation-blocking)
   - [Phase 3: Pairwise Feature Engineering](#phase-3-pairwise-feature-engineering)
   - [Phase 4: ML Training & F0.5 Threshold Tuning (Results: 95.6%)](#phase-4-ml-training--f05-threshold-tuning)
7. [Phase 5: Scaling to the Full Test Dataset (Upcoming)](#7-phase-5-scaling-to-the-full-test-dataset)
8. [Frequently Asked Questions (FAQ for Teammates)](#8-frequently-asked-questions-faq-for-teammates)
9. [Competition Rules & Fair Play Compliance](#9-competition-rules--fair-play-compliance)

---

# 1. What is Entity Resolution? (Layman Explanation)

Imagine three independent retail databases:
* **Source 1:** A verified master registry of businesses.
* **Source 2:** An online delivery app registry.
* **Source 3:** A payment terminal registry.

None of these databases share a common Tax ID, registration number, or database key. In each system, a real-world shop like *McDonald's on 5th Avenue* was typed in by different human operators:
* In Source 1: `"McDonald's", "55 5th Ave, New York, NY", "US"`
* In Source 2: `"MacDonalds Restaurant", "Fifth Avenue #55, NY", "US"`
* In Source 3: `"MCD Fast Food", "Near Central Park, 5th Ave", "US"`

**Entity Resolution (ER)** is the AI task of figuring out that all three of these messy entries are actually the **exact same physical shop**.

---

# 2. Why Simple Solutions Fail: The 5 Types of Data Noise

| # | Noise Pattern | Example in Source 1 | Example in Source 2 / 3 | Why Simple Code Fails | Our Solution |
|---|---|---|---|---|---|
| **1** | **Legal Suffixes** | `Apple Inc.` | `Apple Incorporated` | Text comparison says 0% match | Suffix mapping (`inc` -> `incorporated`) |
| **2** | **Address Abbreviations** | `123 Main St.` | `123 Main Street, Suite 4` | Word count and lengths differ | Levenshtein character edit distance |
| **3** | **Spelling Errors / Typos** | `Starbucks Coffee` | `Sturbucks Coffee` | 1 letter difference breaks exact lookup | Character 3-Grams (`Sta`, `tar`, `arb`) |
| **4** | **Word Transposition** | `Apollo Pharmacy Ltd` | `Pharmacy Apollo` | Words in reverse order | `token_sort_ratio` (sorts words alphabetically first) |
| **5** | **Unseen Countries** | `France` | `France` | Test set has France (train only has US & India) | Relational `is_same_country` feature (0 leakage) |

---

# 3. The Secret to Winning: Mastering the F_0.5 Metric & Singletons

In standard machine learning, models are evaluated on F1 score (equal balance between Precision and Recall). 

In this competition, we are evaluated on **Macro-Averaged F_0.5 Score**:

$$\text{F}_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### What Does This Mean in Plain English?
1. **Precision is weighted 2x more than Recall:**
   * Making a **False Positive** (wrongly merging two different shops) hurts our score **twice as badly** as a False Negative (missing a real match).
   * Therefore, our model must be **extremely conservative and confident** before making a match.
2. **The Power of Singletons:**
   * A "Singleton" is a Source 1 business that has **no matching records** in Source 2 or 3.
   * If our model predicts an empty list for a singleton, we earn a **perfect 1.0 score** for that entity!
   * If our model makes even 1 bad guess for that singleton, our score for that entity drops from **1.0 to 0.0**.
   * By setting a very high prediction threshold (>= 0.95), we harvest maximum points on singletons.

---

# 4. The 2-Stage Master Architecture

Why not just pass every pair directly to a deep neural network or XGBoost?
* Source 1 has ~200,000 entities. Source 2 and 3 have ~400,000 entities combined.
* Comparing all pairs means 200,000 * 400,000 = **80 Billion** comparisons!
* Running a machine learning model on 80 billion pairs would take weeks and crash the computer.

### The Two-Stage Solution:
```
+-------------------------------------------------------------------------+
| STAGE 1: BLOCKING / CANDIDATE GENERATION (Fast & Broad Filter)          |
| • Uses Character 3-Gram TF-IDF + Cosine Similarity                      |
| • Instantly eliminates 99.99% of obvious mismatches                     |
| • Reduces 80 Billion pairs down to 30 candidates per entity             |
| • Output: candidate_pairs.tsv                                           |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
| STAGE 2: MATCHING & CLASSIFICATION (Slow, Precise ML Detective)         |
| • Extracts 6 granular string metrics using C++ RapidFuzz                |
| • LightGBM Model scores candidates from probability 0.00 to 1.00        |
| • Applies strict 0.95 threshold to maximize F_0.5 score                 |
| • Output: matching_results.tsv                                          |
+-------------------------------------------------------------------------+
```

---

# 5. Deep Dive into Every Mathematical Algorithm Used

### A. Character 3-Gram TF-IDF Vectorization
Instead of looking at whole words, we slice names into overlapping 3-letter windows:
* `"Starbucks"` -> `["Sta", "tar", "arb", "rbu", "buc", "uck", "cks"]`
* `"Sturbucks"` -> `["Stu", "tur", "urb", "rbu", "buc", "uck", "cks"]`
* Even with a typo, they share **4 out of 7** 3-grams! This makes typo matching mathematically seamless.

### B. Cosine Similarity & Sparse Matrix Multiplication
Each entity is represented as a high-dimensional mathematical vector A. We compute the angle theta between vectors A and B:

$$\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$

Using `sparse_dot_topn` (a C++ sparse matrix library), we perform this calculation on 300,000 vectors in less than 5 seconds.

### C. The 4 String Distance Algorithms
For each candidate pair, we calculate:
1. **Levenshtein Ratio (`name_ratio` & `address_ratio`):** Minimum number of single-character edits (insertions, deletions, substitutions) to transform string A into string B.
2. **Jaro-Winkler Similarity (`name_jaro`):** A string distance that gives extra mathematical weight to matching prefixes (crucial for brand names like "Target", "Walmart").
3. **Token Sort Ratio (`name_token_sort` & `address_token_sort`):** Splits strings into words, sorts words alphabetically, then compares them.
   * `"Coffee Starbucks"` -> `"Coffee Starbucks"`
   * `"Starbucks Coffee"` -> `"Coffee Starbucks"` -> **100% Match!**
4. **Relational Country Match (`is_same_country`):** `1` if both countries match, `0` otherwise.

### D. How LightGBM Boosted Trees Work
LightGBM builds a series of decision trees that look at all 6 features together:
* *Rule Example:* `IF (address_token_sort > 0.85) AND (name_jaro > 0.90) AND (is_same_country == 1) -> Probability = 0.98 (MATCH)`

---

# 6. Phase-by-Phase Development Log

### Phase 1: Smart Sandboxing & Stratified Sampling (COMPLETED)
* **Code:** `src/01_create_sample.py`
* **What it did:** Sampled 10,000 Source 1 records and guaranteed that all 16,690 matching Source 2 records and 18,078 matching Source 3 records were pulled into `dataset/sample/` along with negative noise.
* **Why:** Allowed us to run the entire pipeline in under 1 minute for rapid development.

---

### Phase 2: Candidate Generation (Blocking) (COMPLETED)
* **Code:** `src/02_blocking.py`
* **What it did:** Built the character 3-gram TF-IDF vectorizer and generated `output/candidate_pairs.tsv` containing the top 30 candidates for each entity.

---

### Phase 3: Pairwise Feature Engineering (COMPLETED)
* **Code:** `src/03_features.py`
* **What it did:** Computed all RapidFuzz C++ similarity features for all 300,000 candidate pairs and created `output/features.csv`.

---

### Phase 4: ML Training & F0.5 Threshold Tuning (COMPLETED)
* **Code:** `src/04_model.py`
* **Model:** LightGBM Binary Classifier.
* **Validation Split:** 80% Train / 20% Holdout Validation grouped by `s1_id`.
* **Feature Importance Ranking:**
  1. `address_token_sort`: **2,360** (Most important feature)
  2. `name_jaro`: **1,829**
  3. `name_token_sort`: **1,595**
  4. `name_ratio`: **1,592**
  5. `address_ratio`: **1,528**
  6. `is_same_country`: **87**

#### Threshold Tuning Results on Validation Set:
```
Threshold 0.10 -> Macro F_0.5: 0.69303
Threshold 0.50 -> Macro F_0.5: 0.92090  (Standard ML cutoff)
Threshold 0.70 -> Macro F_0.5: 0.94184
Threshold 0.80 -> Macro F_0.5: 0.94880
Threshold 0.90 -> Macro F_0.5: 0.95511
Threshold 0.95 -> Macro F_0.5: 0.95605  <<< PEAK SCORE (95.6%) >>>
Threshold 0.98 -> Macro F_0.5: 0.94852
```

---

# 7. Phase 5: Scaling to the Full Test Dataset

When we run Phase 5, we will:
1. Load full `dataset/test/test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv`.
2. Generate test `candidate_pairs.tsv` using the TF-IDF vectorizer.
3. Compute pairwise string features for test candidates.
4. Run inference using our trained LightGBM model with the optimal **`0.95`** threshold.
5. Generate `output/matching_results.tsv`.
6. Run the official validation script:
   ```bash
   python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
7. Package everything into `<team_name>_submission.zip`.

---

# 8. Frequently Asked Questions (FAQ for Teammates)

**Q: How do I run the project on my computer?**
1. Activate virtual environment: `.\venv\Scripts\activate`
2. Run Blocking: `python src/02_blocking.py`
3. Run Features: `python src/03_features.py`
4. Run Model Training: `python src/04_model.py`

**Q: Why didn't we use an LLM (like GPT-4)?**
* Competition rules strictly forbid external APIs.
* Evaluating millions of text pairs with an LLM is too slow and computationally expensive.
* TF-IDF + LightGBM runs in seconds and achieves a 95.6% benchmark.

**Q: What happens if a Source 1 shop doesn't exist in Source 2 or 3?**
* Our model gives all candidates a probability < 0.95, resulting in an empty list. This awards us a perfect **1.0 score** for that singleton.

---

# 9. Competition Rules & Fair Play Compliance

* [x] **No External Data Lookup:** Zero external APIs or geocoding used.
* [x] **Model Size & License:** Open-source LightGBM / Scikit-Learn (<8B parameters).
* [x] **File Formats:** Pure tab-separated `.tsv` with comma-separated IDs.
* [x] **France Support:** Fully supported via relational feature engineering.
