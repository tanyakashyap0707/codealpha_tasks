# CodeAlpha ML Internship - Task 1: Credit Scoring Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

np.random.seed(42)

# Synthetic financial dataset for educational use
n = 1200
income = np.random.randint(15000, 150001, n)
debt = np.random.randint(1000, 90001, n)
payment_history = np.random.randint(0, 11, n)
credit_age = np.random.randint(1, 21, n)
late_payments = np.random.randint(0, 11, n)

score = (
    0.000018 * income
    - 0.000018 * debt
    + 0.55 * payment_history
    + 0.15 * credit_age
    - 0.75 * late_payments
    + np.random.normal(0, 1.2, n)
)
good_credit = (score > 2.8).astype(int)

data = pd.DataFrame({
    "Income": income,
    "Debt": debt,
    "PaymentHistory": payment_history,
    "CreditAgeYears": credit_age,
    "LatePayments": late_payments,
    "GoodCredit": good_credit
})
data.to_csv("credit_data.csv", index=False)

X = data.drop(columns="GoodCredit")
y = data["GoodCredit"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("\n=== Credit Scoring Model ===")
print(classification_report(y_test, pred, target_names=["Poor Credit", "Good Credit"]))
print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Poor", "Good"]
)
plt.title("Credit Scoring - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

RocCurveDisplay.from_predictions(y_test, prob)
plt.title("Credit Scoring - ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# Example prediction
applicant = pd.DataFrame([{
    "Income": 75000,
    "Debt": 18000,
    "PaymentHistory": 8,
    "CreditAgeYears": 7,
    "LatePayments": 1
}])
result = model.predict(applicant)[0]
print("\nExample applicant prediction:",
      "Good Credit" if result == 1 else "Poor Credit")