from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

digits = load_digits()
X, y = digits.data, digits.target

model = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

oob_score = model.oob_score_

cv_scores = cross_val_score(
    RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ),
    X,
    y,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("OOB score:", round(oob_score, 4))
print("Cross-validation 5-fold:", np.round(cv_scores, 4))
print("CV accuracy trung bình:", round(cv_scores.mean(), 4))
print("Độ lệch chuẩn CV:", round(cv_scores.std(), 4))