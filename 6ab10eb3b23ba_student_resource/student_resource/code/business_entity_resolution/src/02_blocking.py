import argparse
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


def clean_text(value):
	if pd.isna(value):
		return ''
	text = str(value).lower()
	text = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text)
	replacements = {'corp': 'corporation', 'ltd': 'limited', 'st': 'street', 'rd': 'road'}
	return ' '.join(replacements.get(token, token) for token in text.split())


def build_candidates(data_dir, output_dir, top_k=30, sim_threshold=0.15):
	os.makedirs(output_dir, exist_ok=True)
	source1 = pd.read_csv(os.path.join(data_dir, 'sample_source1.tsv'), sep='\t', dtype=str).fillna('')
	source2 = pd.read_csv(os.path.join(data_dir, 'sample_source2.tsv'), sep='\t', dtype=str).fillna('')
	source3 = pd.read_csv(os.path.join(data_dir, 'sample_source3.tsv'), sep='\t', dtype=str).fillna('')
	source23 = pd.concat([source2, source3], ignore_index=True)

	source1_text = (source1['business_name'].map(clean_text) + ' ' + source1['business_address'].map(clean_text)).str.strip()
	source23_text = (source23['business_name'].map(clean_text) + ' ' + source23['business_address'].map(clean_text)).str.strip()
	vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, dtype=np.float32)
	source23_matrix = vectorizer.fit_transform(source23_text)
	similarities = sp_matmul_topn(vectorizer.transform(source1_text), source23_matrix.transpose(), top_n=top_k, threshold=sim_threshold)

	rows = []
	for s1_index in range(len(source1)):
		candidate_indices = similarities.getrow(s1_index).indices
		candidate_ids = source23.iloc[candidate_indices]['entity_id'].tolist()
		rows.append({'source1_entity_id': source1.iloc[s1_index]['entity_id'], 'candidate_entity_ids': ','.join(dict.fromkeys(candidate_ids))})
	pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'candidate_pairs.tsv'), sep='\t', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', required=True)
	parser.add_argument('--output-dir', required=True)
	parser.add_argument('--top-k', type=int, default=30)
	parser.add_argument('--sim-threshold', type=float, default=0.15)
	args = parser.parse_args()
	build_candidates(args.data_dir, args.output_dir, args.top_k, args.sim_threshold)
