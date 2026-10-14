"""Train and evaluate the supervised mobile-money fraud classifier."""

import json
from pathlib import Path
import shutil

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data import CATEGORICAL_FEATURES, DATA_PATH, FEATURES, NUMERIC_FEATURES, load_or_generate


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
DOC_ASSETS = ROOT / "docs" / "assets"


def main() -> None:
    frame = load_or_generate(force=True)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["is_fraud"], test_size=0.25, random_state=42, stratify=frame["is_fraud"]
    )
    x_fit, x_validation, y_fit, y_validation = train_test_split(
        x_train, y_train, test_size=0.2, random_state=43, stratify=y_train
    )
    preprocess = ColumnTransformer(
        [
            ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numbers", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_fit, y_fit)
    validation_scores = model.predict_proba(x_validation)[:, 1]
    validation_precision, validation_recall, thresholds = precision_recall_curve(y_validation, validation_scores)
    validation_f1 = (
        2 * validation_precision[:-1] * validation_recall[:-1]
        / (validation_precision[:-1] + validation_recall[:-1] + 1e-12)
    )
    threshold = float(thresholds[validation_f1.argmax()])
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    metrics = {
        "dataset": "synthetic_mobile_money_transactions",
        "rows": int(len(frame)),
        "fraud_rows": int(frame["is_fraud"].sum()),
        "fraud_rate": float(frame["is_fraud"].mean()),
        "test_rows": int(len(y_test)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    DOC_ASSETS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "mobile_money_fraud_model.joblib")
    metrics_path = ARTIFACTS / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored = x_test.copy()
    scored.insert(0, "transaction_id", frame.loc[x_test.index, "transaction_id"])
    scored["actual_is_fraud"] = y_test
    scored["fraud_probability"] = probabilities.round(5)
    scored["predicted_is_fraud"] = predictions
    predictions_path = ARTIFACTS / "test_predictions.csv"
    scored.sort_values("fraud_probability", ascending=False).to_csv(predictions_path, index=False)
    downloads = ROOT / "docs" / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copy2(metrics_path, ROOT / "docs" / "metrics.json")
    shutil.copy2(metrics_path, downloads / "metrics.json")
    shutil.copy2(predictions_path, downloads / "test_predictions.csv")
    shutil.copy2(DATA_PATH, downloads / DATA_PATH.name)

    display = matrix.astype(float)
    fig, axis = plt.subplots(figsize=(5.2, 4.2), layout="constrained")
    image = axis.imshow(display, cmap="YlOrBr")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"], xlabel="Predicted", ylabel="Actual", title="Held-out confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center", color="#241f19", fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(DOC_ASSETS / "confusion-matrix.png", dpi=170, facecolor="#fbf8f1")
    plt.close(fig)
    print(json.dumps({key: metrics[key] for key in ("rows", "fraud_rows", "accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")}, indent=2))
    print(f"Model: {ARTIFACTS / 'mobile_money_fraud_model.joblib'}")
    print(f"Data: {DATA_PATH}")


if __name__ == "__main__":
    main()