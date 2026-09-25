import os
import re
import sys
import gc
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\bcorp\b', 'corporation', text)
    text = re.sub(r'\bltd\b', 'limited', text)
    text = re.sub(r'\bst\b', 'street', text)
    text = re.sub(r'\brd\b', 'road', text)
    return ' '.join(text.split())

def run_full_pipeline(test_dir, output_dir, model_path, chunk_size=50000, top_k=25, sim_threshold=0.15, ml_threshold=0.95):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 65)
    print("? OPTIMIZED FAST FULL TEST INFERENCE PIPELINE")
    print(f"Test Dir: {test_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Blocking Sim Threshold: {sim_threshold} | Top-K: {top_k}")
    print(f"ML Confidence Threshold: {ml_threshold}")
    print("=" * 65, flush=True)
    
    # 1. Load trained LightGBM model
    print("\n[1/4] Loading trained LightGBM model...", flush=True)
    model = joblib.load(model_path)
    
    # 2. Load and index Test Source 2 & Source 3
    print("\n[2/4] Loading and indexing Test Source 2 and Source 3...", flush=True)
    df_s2 = pd.read_csv(os.path.join(test_dir, 'test_source2.tsv'), sep='\t', dtype=str).fillna('')
    df_s3 = pd.read_csv(os.path.join(test_dir, 'test_source3.tsv'), sep='\t', dtype=str).fillna('')
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    
    print(f"Total S2 & S3 entities: {len(df_s23):,}", flush=True)
    
    print("Vectorizing combined text for S2 & S3...", flush=True)
    df_s23['clean_name'] = df_s23['business_name'].apply(clean_text)
    df_s23['clean_addr'] = df_s23['business_address'].apply(clean_text)
    df_s23['combined'] = (df_s23['clean_name'] + ' ' + df_s23['clean_addr']).str.strip()
    
    s23_names = df_s23['clean_name'].values
    s23_addrs = df_s23['clean_addr'].values
    s23_countries = df_s23['country'].str.lower().values
    s23_ids = df_s23['entity_id'].values
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, dtype=np.float32)
    tfidf_s23 = vectorizer.fit_transform(df_s23['combined'])
    print(f"TF-IDF S23 Matrix: {tfidf_s23.shape}", flush=True)
    
    matching_file = os.path.join(output_dir, 'matching_results.tsv')
    candidate_file = os.path.join(output_dir, 'candidate_pairs.tsv')
    
    f_match = open(matching_file, 'w', encoding='utf-8', buffering=1024*1024)
    f_cand = open(candidate_file, 'w', encoding='utf-8', buffering=1024*1024)
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    # 3. Stream Test Source 1
    print("\n[3/4] Streaming Test Source 1 in high-speed batches...", flush=True)
    s1_path = os.path.join(test_dir, 'test_source1.tsv')
    
    total_s1 = 0
    total_matched = 0
    total_singletons = 0
    
    for chunk_idx, df_s1_chunk in enumerate(pd.read_csv(s1_path, sep='\t', dtype=str, chunksize=chunk_size)):
        df_s1_chunk = df_s1_chunk.fillna('')
        n_chunk = len(df_s1_chunk)
        
        df_s1_chunk['clean_name'] = df_s1_chunk['business_name'].apply(clean_text)
        df_s1_chunk['clean_addr'] = df_s1_chunk['business_address'].apply(clean_text)
        df_s1_chunk['combined'] = (df_s1_chunk['clean_name'] + ' ' + df_s1_chunk['clean_addr']).str.strip()
        
        tfidf_s1_chunk = vectorizer.transform(df_s1_chunk['combined'])
        
        # Fast C++ Sparse Dot Top-N with threshold filtering
        matches = sp_matmul_topn(tfidf_s1_chunk, tfidf_s23.transpose(), top_n=top_k, threshold=sim_threshold)
        s1_indices, s23_indices = matches.nonzero()
        
        s1_chunk_names = df_s1_chunk['clean_name'].values
        s1_chunk_addrs = df_s1_chunk['clean_addr'].values
        s1_chunk_countries = df_s1_chunk['country'].str.lower().values
        s1_chunk_ids = df_s1_chunk['entity_id'].values
        
        from collections import defaultdict
        s1_to_s23_map = defaultdict(list)
        for i, j in zip(s1_indices, s23_indices):
            s1_to_s23_map[i].append(j)
            
        # Fast feature extraction
        feature_rows = []
        pair_metadata = []
        
        for s1_loc_idx, s23_idx_list in s1_to_s23_map.items():
            n1 = s1_chunk_names[s1_loc_idx]
            a1 = s1_chunk_addrs[s1_loc_idx]
            c1 = s1_chunk_countries[s1_loc_idx]
            
            for s23_idx in s23_idx_list:
                n2 = s23_names[s23_idx]
                a2 = s23_addrs[s23_idx]
                c2 = s23_countries[s23_idx]
                
                feat = [
                    fuzz.ratio(n1, n2) / 100.0,
                    fuzz.token_sort_ratio(n1, n2) / 100.0,
                    JaroWinkler.normalized_similarity(n1, n2),
                    fuzz.ratio(a1, a2) / 100.0,
                    fuzz.token_sort_ratio(a1, a2) / 100.0,
                    1.0 if (c1 == c2 and c1 != '') else 0.0
                ]
                feature_rows.append(feat)
                pair_metadata.append((s1_loc_idx, s23_idx))
                
        # Fast LightGBM Inference
        matched_results_chunk = defaultdict(list)
        if len(feature_rows) > 0:
            X_chunk = np.array(feature_rows, dtype=np.float32)
            probs = model.predict_proba(X_chunk)[:, 1]
            
            for (s1_loc_idx, s23_idx), p in zip(pair_metadata, probs):
                if p >= ml_threshold:
                    s23_entity_id = s23_ids[s23_idx]
                    matched_results_chunk[s1_loc_idx].append(s23_entity_id)
                    
        # Write batch output
        for loc_idx in range(n_chunk):
            s1_id = s1_chunk_ids[loc_idx]
            
            # Candidates
            cand_s23_indices = s1_to_s23_map.get(loc_idx, [])
            cand_ids = [s23_ids[idx] for idx in cand_s23_indices]
            seen_c = set()
            cand_ids_dedup = [x for x in cand_ids if not (x in seen_c or seen_c.add(x))]
            
            # Matches
            match_ids = matched_results_chunk.get(loc_idx, [])
            seen_m = set()
            match_ids_dedup = [x for x in match_ids if not (x in seen_m or seen_m.add(x))]
            
            f_cand.write(f"{s1_id}\t{','.join(cand_ids_dedup)}\n")
            f_match.write(f"{s1_id}\t{','.join(match_ids_dedup)}\n")
            
            if len(match_ids_dedup) == 0:
                total_singletons += 1
            else:
                total_matched += 1
                
        total_s1 += n_chunk
        print(f"Batch {chunk_idx + 1:02d}: Processed {total_s1:,} S1 entities (Matches: {total_matched:,}, Singletons: {total_singletons:,})", flush=True)
        
    f_match.close()
    f_cand.close()
    
    print("\n" + "=" * 65)
    print("? INFERENCE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total S1 Entities Processed: {total_s1:,}")
    print(f"Entities with Matches: {total_matched:,}")
    print(f"Singletons (Empty Matches): {total_singletons:,}")
    print("=" * 65, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--chunk-size', type=int, default=25000)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--sim-threshold', type=float, default=0.15)
    parser.add_argument('--ml-threshold', type=float, default=0.95)
    args = parser.parse_args()
    run_full_pipeline(args.test_dir, args.output_dir, args.model_path, args.chunk_size, args.top_k, args.sim_threshold, args.ml_threshold)
