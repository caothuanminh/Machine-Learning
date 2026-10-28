import numpy as np
from collections import Counter

X = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1], dtype=int)

bootstrap_samples = [
    [1, 1, 4, 5, 5, 6],
    [2, 3, 3, 4, 6, 6],
    [1, 2, 4, 4, 5, 6]
]

def gini(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return 1 - np.sum(probabilities ** 2)

def best_stump(sample):
    sample = np.array(sample, dtype=float)
    labels = np.array([int(x >= 4) for x in sample])

    candidates = sorted(set(sample))
    best_threshold = None
    best_gini = float("inf")
    best_predictions = None

    for threshold in candidates:
        left = labels[sample <= threshold]
        right = labels[sample > threshold]

        weighted_gini = (
            len(left) / len(labels) * gini(left)
            + len(right) / len(labels) * gini(right)
        )

        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_threshold = threshold
            left_label = Counter(left).most_common(1)[0][0] if len(left) else 0
            right_label = Counter(right).most_common(1)[0][0] if len(right) else 0
            best_predictions = (left_label, right_label)

    return best_threshold, best_gini, best_predictions

predictions = []

for i, sample in enumerate(bootstrap_samples, start=1):
    threshold, score, labels = best_stump(sample)
    prediction = labels[0] if 4.0 <= threshold else labels[1]

    predictions.append(prediction)

    print(f"Mẫu bootstrap {i}: {sample}")
    print(f"  Ngưỡng chia: {threshold}")
    print(f"  Gini tốt nhất: {score:.4f}")
    print(f"  Nhãn nhánh trái: {labels[0]}")
    print(f"  Nhãn nhánh phải: {labels[1]}")
    print(f"  Dự đoán cho x = 4.0: {prediction}")

final_prediction = Counter(predictions).most_common(1)[0][0]

print("\nDự đoán cuối cùng cho x = 4.0:", final_prediction)