import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
import joblib
import argparse

def calculate_f05(precision, recall):
    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate_predictions(df_val, preds, threshold, gt_dict):
    """Computes competition official Macro-averaged F_0.5 score."""
    df_eval = df_val.copy()
    df_eval['pred_prob'] = preds
    df_eval['is_pred_match'] = (df_eval['pred_prob'] >= threshold).astype(int)
    
    # Group predictions by s1_id
    pred_matches = {}
    for s1_id, group in df_eval.groupby('s1_id'):
        matched = group[group['is_pred_match'] == 1]['s23_id'].tolist()
        pred_matches[s1_id] = set(matched)
        
    f05_scores = []
    
    for s1_id, predicted_set in pred_matches.items():
        actual_set = gt_dict.get(s1_id, set())
        
        # Singleton logic according to problem statement:
        if len(actual_set) == 0:
            if len(predicted_set) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        if len(predicted_set) == 0:
            f05_scores.append(0.0)
            continue
            
        tp = len(predicted_set.intersection(actual_set))
        precision = tp / len(predicted_set)
        recall = tp / len(actual_set)
        
        score = calculate_f05(precision, recall)
        f05_scores.append(score)
        
    return np.mean(f05_scores)

def run_model_training(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading features...")
    df = pd.read_csv(os.path.join(output_dir, 'features.csv'))
    
    feature_cols = [
        'name_ratio', 'name_token_sort', 'name_jaro',
        'address_ratio', 'address_token_sort', 'is_same_country'
    ]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Positive matches count: {df['is_match'].sum()} / {len(df)}")
    
    # Group split by s1_id (80% train, 20% validation)
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['s1_id']))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    print(f"Train pairs: {len(df_train)} (Entities: {df_train['s1_id'].nunique()})")
    print(f"Validation pairs: {len(df_val)} (Entities: {df_val['s1_id'].nunique()})")
    
    X_train, y_train = df_train[feature_cols], df_train['is_match']
    X_val, y_val = df_val[feature_cols], df_val['is_match']
    
    # Calculate scale_pos_weight to handle class imbalance
    neg_count = len(y_train) - y_train.sum()
    pos_count = max(1, y_train.sum())
    scale_pos_weight = neg_count / pos_count
    
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    # Feature importance
    print("\n--- Feature Importances ---")
    for feat, imp in zip(feature_cols, model.feature_importances_):
        print(f"  {feat:20s}: {imp}")
        
    print("\nPredicting on Validation Set...")
    val_preds = model.predict_proba(X_val)[:, 1]
    
    # Load Ground Truth for evaluation
    gt = pd.read_csv(os.path.join(data_dir, 'sample_ground_truth.tsv'), sep='\t', dtype=str).fillna('')
    gt_dict = {}
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        matches = [m.strip() for m in str(row['matched_entity_ids']).split(',') if m.strip()]
        gt_dict[s1] = set(matches)
        
    print("\n--- Threshold Tuning for Macro-Averaged F_0.5 ---")
    best_threshold = 0.5
    best_f05 = -1.0
    
    thresholds = [0.1, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
    for th in thresholds:
        score = evaluate_predictions(df_val, val_preds, th, gt_dict)
        print(f"  Threshold {th:0.2f} -> Macro F_0.5: {score:0.5f}")
        if score > best_f05:
            best_f05 = score
            best_threshold = th
            
    print(f"\n>>> OPTIMAL THRESHOLD: {best_threshold:0.2f} with Macro F_0.5: {best_f05:0.5f} <<<")
    
    # Save model
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'models', 'lightgbm_model.pkl'))
    
    # Generate matching_results.tsv for validation set
    df_val_eval = df_val.copy()
    df_val_eval['pred_prob'] = val_preds
    df_val_eval['is_match'] = (df_val_eval['pred_prob'] >= best_threshold).astype(int)
    
    res_dict = {}
    for s1_id, group in df_val_eval.groupby('s1_id'):
        matched = group[group['is_match'] == 1]['s23_id'].tolist()
        res_dict[s1_id] = matched
        
    with open(os.path.join(output_dir, 'matching_results.tsv'), 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in df_val['s1_id'].unique():
            matches = res_dict.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(matches)}\n")
            
    print(f"Sample matching_results.tsv written to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    run_model_training(args.data_dir, args.output_dir)
