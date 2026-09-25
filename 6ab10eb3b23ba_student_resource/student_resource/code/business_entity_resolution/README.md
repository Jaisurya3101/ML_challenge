# Business Entity Resolution Pipeline

Run from this directory with the challenge `student_resource` directory as the data root.

```powershell
python -m pip install -r requirements.txt
python src/01_create_sample.py --train-dir ../../dataset/train --output-dir ../../dataset/sample
python src/02_blocking.py --data-dir ../../dataset/sample --output-dir ../../output
python src/03_features.py
python src/04_model.py
python src/05_full_pipeline.py --test-dir ../../dataset/test --output-dir ../../output --model-path ../../output/models/lightgbm_model.pkl
python ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test --check-ids
```

The sample stage is only for supervised model training. Full-test inference reads every Source 1 record in chunks, generates the final candidate set with character 3-gram TF-IDF top-k blocking, computes six pairwise features, and writes both required TSV outputs. No external data or service is used.
