import argparse
import os
import re
from collections import defaultdict

import pandas as pd


def clean_text(value):
    text = '' if pd.isna(value) else str(value).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    replacements = {'corp': 'corporation', 'ltd': 'limited', 'st': 'street', 'rd': 'road'}
    return ' '.join(replacements.get(token, token) for token in text.split())


def build_index(test_dir, chunk_size=250000):
    by_name = defaultdict(list)
    by_address = defaultdict(list)
    for filename in ('test_source2.tsv', 'test_source3.tsv'):
        path = os.path.join(test_dir, filename)
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=chunk_size):
            chunk = chunk.fillna('')
            names = chunk['business_name'].map(clean_text)
            addresses = chunk['business_address'].map(clean_text)
            countries = chunk['country'].str.lower()
            for entity_id, name, address, country in zip(chunk['entity_id'], names, addresses, countries):
                if name:
                    by_name[(country, name)].append(entity_id)
                if address:
                    by_address[(country, address)].append(entity_id)
    return by_name, by_address


def run(test_dir, output_dir, chunk_size=250000):
    os.makedirs(output_dir, exist_ok=True)
    by_name, by_address = build_index(test_dir, chunk_size)
    matching_path = os.path.join(output_dir, 'matching_results.tsv')
    candidate_path = os.path.join(output_dir, 'candidate_pairs.tsv')
    with open(matching_path, 'w', encoding='utf-8', buffering=1024 * 1024) as matching, open(candidate_path, 'w', encoding='utf-8', buffering=1024 * 1024) as candidates:
        matching.write('source1_entity_id\tmatched_entity_ids\n')
        candidates.write('source1_entity_id\tcandidate_entity_ids\n')
        for chunk in pd.read_csv(os.path.join(test_dir, 'test_source1.tsv'), sep='\t', dtype=str, chunksize=chunk_size):
            chunk = chunk.fillna('')
            names = chunk['business_name'].map(clean_text)
            addresses = chunk['business_address'].map(clean_text)
            countries = chunk['country'].str.lower()
            for entity_id, name, address, country in zip(chunk['entity_id'], names, addresses, countries):
                ids = set(by_name.get((country, name), [])) if name else set()
                if address:
                    ids.update(by_address.get((country, address), []))
                ordered_ids = sorted(ids)
                value = ','.join(ordered_ids)
                candidates.write(f'{entity_id}\t{value}\n')
                matching.write(f'{entity_id}\t{value}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--chunk-size', type=int, default=250000)
    args = parser.parse_args()
    run(args.test_dir, args.output_dir, args.chunk_size)
