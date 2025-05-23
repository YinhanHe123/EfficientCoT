import numpy as np

def evaluate_model(results, dataset):
    """Calculate metrics for the model's performance"""
    metrics = {}
       # If ground truth contains numerical answers, compute numerical accuracy
    data_path = getattr(dataset, 'name', '').lower()
    is_commonsense_qa = 'commonsense_qa' in data_path
    is_coin_flip = 'coin_flip' in data_path
    correct_queries = []

    if is_commonsense_qa:
        # For CommonsenseQA, check if prediction matches the correct answer label (A, B, C, D, E)
        num_correct = 0

        for i, result in enumerate(results):
            prediction = result["prediction"].strip().lower()
            ground_truth = dataset[i]["answer"].strip().lower()

            # Check if the prediction contains or exactly matches the answer option
            # Allow both 'A' and 'A.' as valid formats
            if (ground_truth == prediction or
                ground_truth + '.' == prediction or
                ground_truth in prediction.split() or
                ground_truth + '.' in prediction.split()):
                num_correct += 1
                correct_queries.append(result['query'])

        metrics["numerical_accuracy"] = round(num_correct / len(results), 3) if results else 0
        metrics["correct_queries"] = correct_queries
        return metrics

    elif is_coin_flip:
        # For coin_flip, check if prediction matches yes/no
        num_correct = 0

        for i, result in enumerate(results):
            prediction = result["prediction"].strip().lower()
            ground_truth = dataset[i]["answer"].strip().lower()

            # Check for yes/no match
            if (ground_truth == prediction or
                (ground_truth == 'yes' and ('yes' in prediction.lower() or 'still heads' in prediction.lower())) or
                (ground_truth == 'no' and ('no' in prediction.lower() or 'not heads' in prediction.lower()))):
                num_correct += 1
                correct_queries.append(result['query'])

        metrics["numerical_accuracy"] = round(num_correct / len(results), 3) if results else 0
        metrics["correct_queries"] = correct_queries
        return metrics

    else:
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
                    correct_queries.append(result['query'])

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
            metrics["correct_queries"] = correct_queries

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



