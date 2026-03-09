#  Credit Scoring — Loan Approval Prediction

Classification project to predict whether a bank should approve a loan application based on the applicant's personal and financial characteristics.

---

##  Problem Statement

Classification is a type of supervised machine learning model that produces a binary output — 0 or 1. This project is a binary classification model that, based on historical data, predicts whether a client is reliable and whether a loan should be approved. This has clear practical value: it helps financial institutions save money by making more accurate lending decisions, and reduces the time employees spend on manual credit assessment.
The target variable is `loan_status` (0 = no default, 1 = default). Since classes are imbalanced, `class_weight="balanced"` was applied in Logistic Regression.

---

##  Dataset

- **Source:** [credit_risk_dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Kaggle)
- **Target variable:** `loan_status` (0 = no default, 1 = default)

**Numerical features:**
`person_age`, `person_income`, `person_emp_length`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`

**Categorical features:**
`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`

---

##  Models

| Model | Advantages | Disadvantages |
|-------|------------|---------------|
| Logistic Regression | Simple, interpretable, outputs probability (0–1), fast to train | Assumes linear relationships, struggles with complex patterns |
| Gradient Boosting | High accuracy, handles non-linear patterns, robust to outliers | Slow to train, hard to interpret, more hyperparameters to tune |
| Decision Tree | Very interpretable, no scaling needed | Prone to overfitting, unstable |
| Random Forest | Good accuracy, handles missing values, resistant to overfitting | Less interpretable, slower than single tree |
| SVM | Effective in high dimensions, robust to outliers | Slow on large datasets, hard to interpret |
| KNN | Simple, no training phase | Slow on prediction, sensitive to irrelevant features |
| Neural Network | Extremely powerful, handles any complexity | Needs huge data, very hard to interpret, expensive to train |

### Why these two models?

**Logistic Regression** was chosen as a baseline because its results are easy to interpret — it is straightforward to explain which features influenced the decision and in which direction. Another key advantage is that it outputs a default probability between 0 and 1, giving the bank full flexibility to set its own threshold. For example, a conservative bank can reject clients at a threshold of 0.3, while a risk-tolerant one might set it at 0.7. This allows flexible control over two types of errors: approving a loan for an unreliable client, or rejecting a reliable one.

**Gradient Boosting** was chosen because it is one of the most accurate classification models available. Unlike Logistic Regression, it is not limited to linear relationships and can capture complex patterns in the data. Its higher precision means fewer false predictions — which directly translates to financial savings for the bank. The trade-off is that it is more computationally expensive and harder to interpret, but in a credit scoring context, accuracy is the priority.

Both models use `sklearn.pipeline.Pipeline` with preprocessing: median imputation for missing values and OneHotEncoding for categorical features.


---

##  Results

| Metric      | Logistic Regression | Gradient Boosting |
|-------------|---------------------|-------------------|
| Accuracy    | 0.868               | 0.921             |
| Precision   | 0.769               | 0.918             |
| Recall      | 0.564               | 0.699             |
| F1-score    | 0.651               | 0.794             |
| ROC-AUC     | 0.869               | 0.925             |
| Log-loss    | 0.339               | 0.242             |

**Gradient Boosting outperforms Logistic Regression across all metrics without exception.**

Accuracy is higher (0.921 vs 0.868), F1-score is better (0.794 vs 0.651), and log-loss is lower (0.242 vs 0.339), indicating more confident and well-calibrated probabilities. The gap in Recall is especially notable — 0.699 vs 0.564: Gradient Boosting catches significantly more real defaults, which is critical for a bank, since a missed default costs more than an unnecessary rejection.

Gradient Boosting also leads on both curve-based metrics: ROC-AUC of 0.925 vs 0.869 means better separation between reliable and unreliable clients, and the gap in Precision-Recall (0.87 vs 0.72) shows that GB identifies rare defaults more precisely — which is the most important thing in credit scoring.

That said, Logistic Regression with AUC 0.87 is far from a poor result. A reasonable approach would be to use Gradient Boosting as the primary model, while using Logistic Regression to explain decisions to regulators and clients — since its outputs are much easier to interpret.

Gradient Boosting is harder to explain and less transparent, but it catches significantly more real defaults — and in lending, that matters more than clean reporting. Every missed default is a direct financial loss. Therefore, despite interpretability challenges, its superiority in Recall and F1 makes it the preferred choice for this task.

---

##  Project Structure

```
credit-scoring/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   └── exploration.ipynb
├── results/
│   └── metrics.png
└── src/
    ├── logreg.py
    └── gradient.py
```

---

## ⚙️ How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/credit-scoring.git
cd credit-scoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the models:
```bash
python logistic_regression.py
python gradient_boosting.py
```

---

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib

---

## 👤 Author

**Zheleikina Anna**  
[LinkedIn](https://www.linkedin.com/in/anna-zheleikina-136094291?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BejQQ0VslSbyNOjuq8ainzw%3D%3D) | [GitHub](https://github.com/aicezhe)