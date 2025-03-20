import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(results, dataset):
    """Calculate metrics for the model's performance"""
    metrics = {}
       # If ground truth contains numerical answers, compute numerical accuracy
    num_correct = 0
    num_close = 0  # Within 1% error
    relative_errors = []

    for i, result in enumerate(results):
        try:
            # Extract all numbers from the prediction
            pred_nums = extract_all_numbers(result["prediction"])

            # Get the answer from the data
            gt_num = float(dataset[i]["answer"])

            # Check for exact match among any of the extracted numbers
            if any(abs(pred - gt_num) < 1e-6 for pred in pred_nums):  # Allow for small floating point differences
                num_correct += 1

            # Find the closest match and calculate relative error
            if pred_nums:
                closest_pred = min(pred_nums, key=lambda x: abs(x - gt_num))
                rel_error = abs(closest_pred - gt_num) / abs(gt_num)
                relative_errors.append(rel_error)

                # Check if the closest match is within 1% error
                if rel_error < 0.01:
                    num_close += 1
        except (ValueError, TypeError):
            continue

        if results:  # Avoid division by zero
            metrics["numerical_accuracy"] = round(num_correct / len(results), 3)
            metrics["close_match_rate"] = round(num_close / len(results), 3)

        if relative_errors:
            metrics["mean_relative_error"] = np.mean(relative_errors)
            metrics["median_relative_error"] = np.median(relative_errors)

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

def extract_all_numbers(text):
    """Extract all numbers from a text"""
    import re
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return [float(num) for num in numbers]