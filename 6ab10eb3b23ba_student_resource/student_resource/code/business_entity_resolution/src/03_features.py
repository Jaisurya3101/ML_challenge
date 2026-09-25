import pandas as pd
import numpy as np
import os
import re
import argparse
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm import tqdm


def clean_text(value):
    if pd.isna(value):
        return ''
    text = re.sub(r'[^a-z0-9\s]', ' ', str(value).lower())
    replacements = {'corp': 'corporation', 'ltd': 'limited', 'st': 'street', 'rd': 'road'}
    return ' '.join(replacements.get(token, token) for token in text.split())

def compute_features(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, 'sample_source1.tsv'), sep='\t', dtype=str).fillna('')
    df_s2 = pd.read_csv(os.path.join(data_dir, 'sample_source2.tsv'), sep='\t', dtype=str).fillna('')
    df_s3 = pd.read_csv(os.path.join(data_dir, 'sample_source3.tsv'), sep='\t', dtype=str).fillna('')
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s23.set_index('entity_id').to_dict('index')
    
    print("Loading candidates...")
    candidates = pd.read_csv(os.path.join(output_dir, 'candidate_pairs.tsv'), sep='\t', dtype=str).fillna('')
    
    pairs = []
    for _, row in candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        for c in cands:
            if c:
                pairs.append({'s1_id': s1_id, 's23_id': c})
                
    df_pairs = pd.DataFrame(pairs)
    print(f"Total candidate pairs to process: {len(df_pairs)}")
    
    print("Loading Ground Truth...")
    gt = pd.read_csv(os.path.join(data_dir, 'sample_ground_truth.tsv'), sep='\t', dtype=str).fillna('')
    true_pairs = set()
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
        for m in matches:
            if m:
                true_pairs.add(f"{s1}_{m}")
                
    print("Calculating features...")
    
    features_list = []
    for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        s1_id = row['s1_id']
        s23_id = row['s23_id']
        
        s1_data = s1_dict.get(s1_id, {})
        s23_data = s23_dict.get(s23_id, {})
        
        n1 = clean_text(s1_data.get('business_name', ''))
        n2 = clean_text(s23_data.get('business_name', ''))
        
        a1 = clean_text(s1_data.get('business_address', ''))
        a2 = clean_text(s23_data.get('business_address', ''))
        
        c1 = str(s1_data.get('country', '')).lower()
        c2 = str(s23_data.get('country', '')).lower()
        
        is_match = 1 if f"{s1_id}_{s23_id}" in true_pairs else 0
        
        features = {
            's1_id': s1_id,
            's23_id': s23_id,
            'name_ratio': fuzz.ratio(n1, n2) / 100.0,
            'name_token_sort': fuzz.token_sort_ratio(n1, n2) / 100.0,
            'name_jaro': JaroWinkler.normalized_similarity(n1, n2),
            'address_ratio': fuzz.ratio(a1, a2) / 100.0,
            'address_token_sort': fuzz.token_sort_ratio(a1, a2) / 100.0,
            'is_same_country': 1 if (c1 == c2 and c1 != '') else 0,
            'is_match': is_match
        }
        features_list.append(features)
        
    df_features = pd.DataFrame(features_list)
    out_path = os.path.join(output_dir, 'features.csv')
    df_features.to_csv(out_path, index=False)
    print(f"Features saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    compute_features(args.data_dir, args.output_dir)
