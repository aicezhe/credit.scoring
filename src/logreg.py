import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, log_loss,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt



df = pd.read_csv(r"C:\Users\USER\Desktop\reviews analisis\нн\credit_risk_dataset.csv")



num_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]


cat_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]



X = df[num_cols + cat_cols]
y = df["loan_status"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
])


pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
])



pipeline.fit(X_train, y_train)



y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print(f"Log-loss:  {log_loss(y_test, y_proba):.3f}")



plt.figure()
RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("ROC curve - Logistic Regression")
plt.grid(True, alpha=0.3)
plt.savefig('results/roc_logreg.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure()
PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test)
plt.title("Precision-Recall - Logistic Regression")
plt.grid(True, alpha=0.3)
plt.savefig('results/pr_logreg.png', dpi=150, bbox_inches='tight')
plt.show()


