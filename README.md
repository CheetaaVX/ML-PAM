# ML-PAM — NLP-Based Privileged Access Management

ML-PAM is an NLP-based machine learning project that classifies corporate email access requests as **trusted** or **suspicious**. It also explores adversarial text manipulation using HotFlip to evaluate model robustness.

## Features

- NLP-based email access-request classification
- Logistic Regression model training
- Separate processing of email body, subject, and metadata
- Model evaluation using classification metrics and ROC-AUC
- Model serialization using Joblib
- Adversarial NLP testing using HotFlip

## Dataset

**Dataset:** `corporate_mail_access_requests_250k_balanced.csv`

- Total records: 250,000
- Total columns: 22
- Classification labels: `trusted` and `suspicious`

```python
df = pd.read_csv(
    "/content/sample_data/corporate_mail_access_requests_250k_balanced.csv"
)

print(df.shape)
```

## Model Training

The notebook trains a Logistic Regression classifier using text and metadata features.

The trained model and preprocessing components are saved separately for reuse.

```python
joblib.dump(model, "model.pkl")

joblib.dump(body_vectorizer, "body_vectorizer.pkl")

joblib.dump(subject_vectorizer, "subject_vectorizer.pkl")

joblib.dump(metadata_preprocessor, "metadata_preprocessor.pkl")
```

### Saved Files

| File | Purpose |
|---|---|
| `model.pkl` | Trained classification model |
| `body_vectorizer.pkl` | Email body text processing |
| `subject_vectorizer.pkl` | Email subject text processing |
| `metadata_preprocessor.pkl` | Metadata preprocessing |

## Model Evaluation

### Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Trusted | 0.90 | 0.95 | 0.93 |
| Suspicious | 0.95 | 0.89 | 0.92 |
| **Overall accuracy** | | | **0.92** |

- **Accuracy:** 92%
- **Macro F1-score:** 0.92
- **Weighted F1-score:** 0.92
- **ROC-AUC:** 0.96293

## Disclaimer

This project is intended for educational purposes and authorized security research. NLP predictions should not independently authorize privileged access; authorization must be enforced through separate security policies.

