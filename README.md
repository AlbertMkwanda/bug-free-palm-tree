# Mobile Money Transaction Fraud Detection

**Student:** Albert Mkwanda (serial no. 1, registration no. 23311351001)  
**Approach:** Supervised classification.

This project trains a Random Forest classifier to distinguish fraudulent from legitimate Malawi-style mobile-money transactions. It generates a deterministic 6,000-row synthetic dataset, performs stratified train/validation/test splits, selects a probability threshold by validation F1 using training data only, and reports accuracy, precision, recall, F1, ROC-AUC, average precision, and the untouched test confusion matrix.

## Data and scope

`data/mobile_money_transactions.csv` is generated locally by `src/data.py` with seed 42. All accounts, transaction values, and labels are synthetic; they are not customer records, and the measured performance is not a claim about any real provider. The label is sampled from a defined synthetic risk process. Identifier columns are retained for review but excluded from model features. The two ZIP files in the parent folder contain SMS/token and scam-message text, not mobile-money transaction records, so they are not used as training data.

## Run

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train
```

Training regenerates the dataset if absent and writes `artifacts/mobile_money_fraud_model.joblib`, `artifacts/metrics.json`, `artifacts/test_predictions.csv`, and the portfolio confusion-matrix image.

Score a CSV containing all feature columns listed in `src/data.py`:

```powershell
.\.venv\Scripts\python.exe -m src.predict path\to\transactions.csv --output scored.csv
```

## Portfolio and submission

The three-page portfolio is in `docs/` (`index.html`, `methodology.html`, and `results.html`). The included GitHub Actions workflow publishes `docs/` to GitHub Pages after the project is pushed to a `main` branch and Pages is enabled for GitHub Actions. Before submission, create the GitHub repository, replace the portfolio's repository placeholder, add `stanslausphillimon@gmail.com` as a collaborator, push the project, and record the resulting repository and Pages URLs. Publishing and collaborator changes require the owner's GitHub account and are not performed from this local workspace.

## Limitations

Synthetic labels make this a reproducible coursework demonstration, not a deployment-ready fraud system. Real use would require consented, provider-approved, representative transaction data, temporal validation, threshold calibration with investigators, privacy/security controls, and human review before adverse action.