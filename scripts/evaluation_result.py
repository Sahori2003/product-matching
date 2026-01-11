# ===============================
#  EVALUATION.PY
# ===============================

import pandas as pd
import numpy as np
from tqdm import tqdm

# -----------------------------
# CORE METRICS
# -----------------------------

def top_k_accuracy(predictions, ground_truth, k=1):
    """Calculate Top-K Accuracy."""
    correct = 0
    for pred_list, true_sku in zip(predictions, ground_truth):
        if true_sku in pred_list[:k]:
            correct += 1
    return correct / len(ground_truth)


def precision_recall_f1(predictions, ground_truth, k=1):
    """Calculate Precision, Recall, and F1 Score for Top-K."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for pred_list, true_sku in zip(predictions, ground_truth):
        top_k_preds = pred_list[:k]
        
        if true_sku in top_k_preds:
            true_positives += 1
        else:
            false_negatives += 1
            false_positives += k  # all k predictions are wrong
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


# -----------------------------
# EVALUATION FUNCTION
# -----------------------------

def evaluate_model(search_function, test_data, k=10):
    """
    Evaluate the model on test data.
    
    Parameters:
    -----------
    search_function : function
        Function that takes (alias, k) and returns search results
    test_data : DataFrame
        Must have columns: ['alias', 'true_sku']
    k : int
        Number of top results to retrieve
    
    Returns:
    --------
    dict : Evaluation metrics
    """
    
    print("Running predictions...")
    predictions = []
    ground_truth = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        alias = row['alias']
        true_sku = row['SKU_Name']
        
        # Get predictions
        results = search_function(alias, k=k)
        pred_skus = [r['sku'] for r in results['results']]
        
        predictions.append(pred_skus)
        ground_truth.append(true_sku)
    
    print("Calculating metrics...\n")
    
    # Top-1 Accuracy
    top1_acc = top_k_accuracy(predictions, ground_truth, k=1)
    
    # Top-10 Accuracy
    top10_acc = top_k_accuracy(predictions, ground_truth, k=10)
    
    # Precision, Recall, F1 for Top-1
    precision_top1, recall_top1, f1_top1 = precision_recall_f1(predictions, ground_truth, k=1)
    
    # Precision, Recall, F1 for Top-10
    precision_top10, recall_top10, f1_top10 = precision_recall_f1(predictions, ground_truth, k=10)
    
    # Print Results
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Samples: {len(ground_truth)}\n")
    
    print("TOP-1 METRICS:")
    print(f"  Accuracy  : {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Precision : {precision_top1:.4f}")
    print(f"  Recall    : {recall_top1:.4f}")
    print(f"  F1 Score  : {f1_top1:.4f}\n")
    
    print("TOP-10 METRICS:")
    print(f"  Accuracy  : {top10_acc:.4f} ({top10_acc*100:.2f}%)")
    print(f"  Precision : {precision_top10:.4f}")
    print(f"  Recall    : {recall_top10:.4f}")
    print(f"  F1 Score  : {f1_top10:.4f}")
    print("=" * 50)
    
    return {
        'top1_accuracy': top1_acc,
        'top10_accuracy': top10_acc,
        'top1_precision': precision_top1,
        'top1_recall': recall_top1,
        'top1_f1': f1_top1,
        'top10_precision': precision_top10,
        'top10_recall': recall_top10,
        'top10_f1': f1_top10,
        'predictions': predictions,
        'ground_truth': ground_truth
    }


# -----------------------------
# EXAMPLE USAGE
# -----------------------------

if __name__ == "__main__":
    # Import your search function
    from faiss_index_builder import search_alias
    
    # Load test data
    test_df = pd.read_excel(r"C:/Users/raad2/Downloads/Product Matching/data/Cleaned_data.xlsx")  # needs columns: alias, true_sku
    
    # Run evaluation
    results = evaluate_model(search_alias, test_df, k=10)
    
    # Access specific metrics
    print(f"\nTop-1 Accuracy: {results['top1_accuracy']:.4f}")
    print(f"Top-1 F1 Score: {results['top1_f1']:.4f}")