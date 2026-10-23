```bash
git clone <repository-url>
cd fake-review-detector
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```css
pip install -r requirements.txt
python train_model.py --data data/sample_reviews.csv
streamlit run app.py
```

# Fake Review Detection System using Machine Learning

A polished, intermediate-level Streamlit project for screening customer reviews as **Likely Genuine** or **Suspicious Review**. It uses review language to prioritise items for human review and deliberately does not claim to prove fraud.

## Features

- Modern dark-navy Streamlit dashboard with purple and cyan accents.
- Individual review analysis with confidence, cautious plain-language guidance, and Logistic Regression feature contributions when available.
- Bulk CSV analysis with preview, validation, predictions, Plotly pie/bar charts, and downloadable enriched results.
- Safe preprocessing: lowercasing, URL removal, punctuation cleanup, whitespace normalisation, NLTK tokenisation, and preserved negations such as `not`.
- Combined word (1–2) and character (3–5) TF-IDF n-grams.
- Training comparison between Multinomial Naive Bayes, Logistic Regression, and calibrated Linear SVM.
- Accuracy, precision, recall, macro F1-score, and confusion matrix reporting.
- Automatic selection and joblib persistence of the best holdout macro-F1 model; Logistic Regression breaks exact ties and is the default baseline.

## Folder structure

```text
fake-review-detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_reviews.csv
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── prediction.py
│   ├── explainability.py
│   └── utils.py
└── tests/
    └── test_preprocessing.py
```

After training, `models/` also contains `review_model.joblib`, `tfidf_vectorizer.joblib`, and `model_metrics.json`. These generated files are intentionally not included in the starter project.

## Dataset format

Training CSVs must contain these columns:

| Column | Required | Description |
| --- | --- | --- |
| `review_text` | Yes | The review body to analyse. |
| `label` | Yes | `fake`/`genuine`, or binary `1`/`0`. `1` means suspicious/fake. |
| `rating` | No | Star rating metadata, typically 1–5. The baseline model uses text features only. |

Example:

```csv
review_text,label,rating
"The item worked well after a week of use.",genuine,4
"Perfect miracle product, everyone must buy now!",fake,5
```

`data/sample_reviews.csv` is intentionally tiny and illustrative. It is useful only for verifying that the app and training command work; it is not a valid dataset for evaluating a real-world fraud-detection system.

## Model-training process

1. The training script validates required columns, safely cleans review text, and normalises supported labels to `0` (genuine) and `1` (suspicious).
2. It stratifies the data into training and holdout subsets.
3. A `FeatureUnion` produces word-level and character-level TF-IDF features.
4. Naive Bayes, Logistic Regression, and a calibrated Linear SVM are trained on exactly the same features.
5. The classifier with the best macro F1-score is saved with the fitted vectorizer using `joblib`. Exact ties use Logistic Regression as the default baseline.
6. Metrics and the confusion matrix are written to `models/model_metrics.json` and shown in the app's **Model Insights** page.

## Evaluation metrics

- **Accuracy:** overall share of correct holdout predictions.
- **Precision:** when a review is flagged suspicious, how often that flag was correct in the labelled holdout data.
- **Recall:** how many labelled suspicious reviews were identified.
- **Macro F1-score:** the balanced mean F1 over both classes. This is used for model selection because it helps reduce dominance by a larger class.
- **Confusion matrix:** a count of genuine/suspicious predictions against the labelled holdout set.

Run the small unit test suite with:

```bash
pytest
```

## Screenshots

Add screenshots of your customised deployment here:

- `screenshots/dashboard.png` — Home dashboard
- `screenshots/single-review.png` — Individual assessment
- `screenshots/bulk-analysis.png` — Batch results and charts

## Limitations and ethical disclaimer

This project is a screening and learning tool, not proof that any person, business, or review is fraudulent. Language models can be inaccurate, biased, or poorly calibrated on new products, communities, languages, and writing styles. Never use the output as the sole reason to remove content, penalise a user, or make a high-impact decision. Use representative, lawfully collected labelled data, evaluate across relevant groups, and keep a human reviewer in the final loop.

## Future improvements

- Add a larger, carefully labelled and de-duplicated dataset.
- Measure calibration, class imbalance, subgroup performance, and drift over time.
- Add reviewer-level and product-level signals only with appropriate privacy safeguards.
- Support language detection and separately evaluated multilingual models.
- Add an annotation workflow so human feedback can be reviewed before retraining.
