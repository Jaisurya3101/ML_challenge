import argparse
import os
import pandas as pd


def create_sample(train_dir, output_dir, source1_limit=10000, random_state=42):
	os.makedirs(output_dir, exist_ok=True)
	source1 = pd.read_csv(os.path.join(train_dir, 'train_source1.tsv'), sep='\t', dtype=str).fillna('')
	source2 = pd.read_csv(os.path.join(train_dir, 'train_source2.tsv'), sep='\t', dtype=str).fillna('')
	source3 = pd.read_csv(os.path.join(train_dir, 'train_source3.tsv'), sep='\t', dtype=str).fillna('')
	ground_truth = pd.read_csv(os.path.join(train_dir, 'train_ground_truth.tsv'), sep='\t', dtype=str).fillna('')

	sample_s1 = source1.sample(n=min(source1_limit, len(source1)), random_state=random_state)
	selected_s1 = set(sample_s1['entity_id'])
	selected_ids = set()
	for _, row in ground_truth[ground_truth['source1_entity_id'].isin(selected_s1)].iterrows():
		selected_ids.update(x.strip() for x in row['matched_entity_ids'].split(',') if x.strip())

	sample_s2 = source2[source2['entity_id'].isin(selected_ids)].copy()
	sample_s3 = source3[source3['entity_id'].isin(selected_ids)].copy()
	sample_s2 = pd.concat([sample_s2, source2[~source2['entity_id'].isin(selected_ids)].sample(
		n=min(len(source2) - len(sample_s2), len(sample_s1)), random_state=random_state)], ignore_index=True)
	sample_s3 = pd.concat([sample_s3, source3[~source3['entity_id'].isin(selected_ids)].sample(
		n=min(len(source3) - len(sample_s3), len(sample_s1)), random_state=random_state)], ignore_index=True)
	sample_gt = ground_truth[ground_truth['source1_entity_id'].isin(selected_s1)].copy()

	sample_s1.to_csv(os.path.join(output_dir, 'sample_source1.tsv'), sep='\t', index=False)
	sample_s2.to_csv(os.path.join(output_dir, 'sample_source2.tsv'), sep='\t', index=False)
	sample_s3.to_csv(os.path.join(output_dir, 'sample_source3.tsv'), sep='\t', index=False)
	sample_gt.to_csv(os.path.join(output_dir, 'sample_ground_truth.tsv'), sep='\t', index=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train-dir', required=True)
	parser.add_argument('--output-dir', required=True)
	parser.add_argument('--source1-limit', type=int, default=10000)
	args = parser.parse_args()
	create_sample(args.train_dir, args.output_dir, args.source1_limit)
