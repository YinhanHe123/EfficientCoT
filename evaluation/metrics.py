import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(results, dataset):
    """Calculate metrics for the model's performance"""
    metrics = {}

    # Exact match accuracy
    correct = 0
    for result in results:
        if result["prediction"].strip() == result["ground_truth"].strip():
            correct += 1

    metrics["exact_match"] = correct / len(results)

    # If ground truth contains numerical answers, compute numerical accuracy
    if all("ground_truth" in result and is_number(result["ground_truth"]) for result in results):
        num_correct = 0
        num_close = 0  # Within 1% error
        relative_errors = []

        for result in results:
            try:
                pred_num = extract_number(result["prediction"])
                gt_num = extract_number(result["ground_truth"])

                # Exact match
                if abs(pred_num - gt_num) < 1e-6:  # Allow for small floating point differences
                    num_correct += 1

                # Close match (within 1% error)
                if gt_num != 0:
                    rel_error = abs(pred_num - gt_num) / abs(gt_num)
                    relative_errors.append(rel_error)
                    if rel_error < 0.01:
                        num_close += 1
            except (ValueError, TypeError):
                continue

        metrics["numerical_accuracy"] = num_correct / len(results)
        metrics["close_match_rate"] = num_close / len(results)

        if relative_errors:
            metrics["mean_relative_error"] = np.mean(relative_errors)
            metrics["median_relative_error"] = np.median(relative_errors)

    # Check if we can perform binary classification metrics
    # For example, if ground truth and predictions can be mapped to binary values
    try:
        binary_gt = [int(bool(extract_number(r["ground_truth"]) > 0)) for r in results]
        binary_pred = [int(bool(extract_number(r["prediction"]) > 0)) for r in results]

        metrics["accuracy"] = accuracy_score(binary_gt, binary_pred)
        metrics["precision"] = precision_score(binary_gt, binary_pred, zero_division=0)
        metrics["recall"] = precision_score(binary_gt, binary_pred, zero_division=0)
        metrics["f1"] = f1_score(binary_gt, binary_pred, zero_division=0)
    except:
        # Skip binary metrics if not applicable
        pass

    return metrics

def is_number(s):
    """Check if a string represents a number"""
    try:
        float(s.strip())
        return True
    except (ValueError, TypeError):
        return False

def extract_number(text):
    """Extract the first number from a text"""
    # Try to find a number in the text
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return float(numbers[0])
    raise ValueError("No number found in text")