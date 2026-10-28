import numpy as np

X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([-1, -1, -1, 1, 1, 1, 1, 1], dtype=int)

def train_stump(X, y, weights):
    thresholds = (X[:-1] + X[1:]) / 2
    best_error = float("inf")
    best_stump = None

    for threshold in thresholds:
        for polarity in [1, -1]:
            predictions = np.ones(len(X), dtype=int)

            if polarity == 1:
                predictions[X <= threshold] = -1
            else:
                predictions[X > threshold] = -1

            error = np.sum(weights[predictions != y])

            if error < best_error:
                best_error = error
                best_stump = (threshold, polarity, predictions.copy())

    return best_stump, best_error

def predict_stump(X, stump):
    threshold, polarity, _ = stump
    predictions = np.ones(len(X), dtype=int)

    if polarity == 1:
        predictions[X <= threshold] = -1
    else:
        predictions[X > threshold] = -1

    return predictions

def adaboost(X, y, T=3):
    n = len(X)
    weights = np.ones(n) / n
    learners = []

    for t in range(T):
        stump, error = train_stump(X, y, weights)
        error = np.clip(error, 1e-10, 1 - 1e-10)

        alpha = 0.5 * np.log((1 - error) / error)
        predictions = predict_stump(X, stump)

        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)

        learners.append((stump, alpha))

        print(f"Vòng {t + 1}")
        print("Ngưỡng:", stump[0])
        print("Polarity:", stump[1])
        print("Weighted error:", round(error, 4))
        print("Alpha:", round(alpha, 4))
        print("Trọng số mẫu:", np.round(weights, 4))
        print()

    return learners

def predict_adaboost(X, learners):
    total = np.zeros(len(X))

    for stump, alpha in learners:
        total += alpha * predict_stump(X, stump)

    return np.where(total >= 0, 1, -1)

learners = adaboost(X, y, T=3)

final_predictions = predict_adaboost(X, learners)

print("Nhãn thật:", y)
print("Nhãn dự đoán:", final_predictions)
print("Độ chính xác:", np.mean(final_predictions == y))