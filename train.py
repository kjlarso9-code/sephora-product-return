# ======================================================================
# Hands-On 3 — CLASSIFICATION (Sephora Product Return Risk)
# Neural Network (MLPClassifier) + SVC (compact sweeps, fast)
# Logs to Databricks MLflow (fallback to local if creds not present)
# Metrics: Accuracy, F1, ROC-AUC
# User: <kjlarso9>@asu.edu
# ======================================================================

import os
import io
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance


plt.rcParams["figure.dpi"] = 120
mlflow.sklearn.autolog(log_models=True)

USER_EMAIL = "kjlarso9@asu.edu"
EXP_NAME = "SephoraProductReturnRisk_Classification"
TARGET = "high_return_risk"
CSV_PATH = "sephora_website_dataset.csv"  # CSV must be in the repo next to this file


def setup_mlflow():
    """
    Configure MLflow to talk to Databricks using environment variables:
    DATABRICKS_HOST and DATABRICKS_TOKEN.
    """
    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")

    if not databricks_host or not databricks_token:
        print(
            "Warning: DATABRICKS_HOST or DATABRICKS_TOKEN not set in environment.\n"
            "MLflow tracking to Databricks may fail. Please configure these env vars."
        )
    else:
        print("✅ Databricks credentials found in environment.")

    mlflow.set_tracking_uri("databricks")

    experiment_name = f"/Users/{USER_EMAIL}/{EXP_NAME}"
    mlflow.set_experiment(experiment_name)
    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experiment:", experiment_name)

    return experiment_name


def robust_read_csv_local(path: str) -> pd.DataFrame:
    """Reads local CSV with auto-detected separator; handles semicolons; cleans headers."""
    print(f"📁 Loading Sephora CSV from: {path}")
    with open(path, "rb") as f:
        raw = f.read()

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        df = pd.read_csv(io.BytesIO(raw))

    if df.shape[1] == 1 and ";" in df.columns[0]:
        df = pd.read_csv(io.BytesIO(raw), sep=";", engine="python")

    df.columns = [c.strip().strip('"') for c in df.columns]
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    return df


def engineer_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target Engineering: "High return / dislike risk"
      - Use 'rating' column
      - Drop rows with missing or 0 rating (0 often means 'unrated')
      - high_risk = 1 if rating < 3, else 0
    """
    if "rating" not in df.columns:
        raise ValueError("Expected a 'rating' column in the Sephora dataset.")

    df = df.copy()
    df = df[~df["rating"].isna()]  # drop missing ratings
    df = df[df["rating"] > 0]  # drop '0' ratings (unrated / noise)

    df[TARGET] = (df["rating"] < 3.0).astype(int)
    print(df[TARGET].value_counts())
    return df


def build_features(df: pd.DataFrame):
    """
    Feature selection and preprocessing.
    """
    drop_cols = [
        "id",
        "rating",
        "URL",
        "details",
        "how_to_use",
        "MarketingFlags_content",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df[TARGET]
    X = df.drop(columns=[TARGET] + drop_cols)

    # Cast numerics to float64 (prevents MLflow schema warnings)
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    print(f"Numeric cols: {len(num_cols)} -> {num_cols}")
    print(f"Categorical cols: {len(cat_cols)} -> {cat_cols}")

    prep = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, y, prep, num_cols, cat_cols


def log_confusion_matrix_plot(y_true, y_pred, fname="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    mlflow.log_artifact(fname)


def log_prob_hist_plot(y_prob, fname="prob_hist.png"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(y_prob, bins=20)
    ax.set_title("Predicted High-Risk Probability")
    ax.set_xlabel("P(high_return_risk=1)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    mlflow.log_artifact(fname)


def feat_names_from_preprocessor(prep, num_cols, cat_cols):
    names = list(num_cols)
    if len(cat_cols):
        ohe = prep.named_transformers_["cat"]
        names += ohe.get_feature_names_out(cat_cols).tolist()
    return names


def perm_top_for_best(
    pipe, X_eval, y_eval, names, family_prefix, n_repeats=3, max_rows=4000
):
    """Fast permutation importance on a subsample; logs top-20 text artifact."""
    if len(X_eval) > max_rows:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X_eval), size=max_rows, replace=False)
        X_eval, y_eval = X_eval.iloc[idx], y_eval.iloc[idx]

    perm = permutation_importance(
        pipe,
        X_eval,
        y_eval,
        n_repeats=n_repeats,
        random_state=42,
        scoring="f1",
    )
    imps = perm.importances_mean
    top = int(np.argmax(imps))
    top_name, top_val = names[top], float(imps[top])

    order = np.argsort(imps)[::-1]
    lines = ["rank,feature,importance_mean"]
    for r, j in enumerate(order[:20], start=1):
        lines.append(f"{r},{names[j]},{imps[j]:.8f}")

    fname = f"{family_prefix}_perm_top20.txt"
    with open(fname, "w") as f:
        f.write("\n".join(lines))
    mlflow.log_artifact(fname)

    return top_name, top_val


def run_nn(X_train, X_test, y_train, y_test, prep, num_cols, cat_cols):
    nn_grid = [
        {"hidden_layer_sizes": (64,), "learning_rate_init": 1e-3, "alpha": 1e-4},
        {"hidden_layer_sizes": (64, 32), "learning_rate_init": 5e-4, "alpha": 1e-4},
        {"hidden_layer_sizes": (64, 32), "learning_rate_init": 5e-4, "alpha": 1e-3},
    ]
    best_nn = {"f1": -np.inf, "run_id": None, "pipe": None}

    for p in nn_grid:
        name = f"NNCls_h{p['hidden_layer_sizes']}_lr{p['learning_rate_init']}_alpha{p['alpha']}"
        with mlflow.start_run(run_name=name) as run:
            model = MLPClassifier(
                hidden_layer_sizes=p["hidden_layer_sizes"],
                learning_rate_init=p["learning_rate_init"],
                alpha=p["alpha"],
                max_iter=250,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            )
            pipe = Pipeline([("prep", prep), ("clf", model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred))
            roc = float(roc_auc_score(y_test, y_proba))

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc)

            mlflow.set_tag("primary_metric", "f1")
            mlflow.set_tag("primary_metric_goal", "maximize")
            mlflow.set_tag("model_family", "MLPClassifier")

            with open("run_metrics.json", "w") as f:
                json.dump({"accuracy": acc, "f1": f1, "roc_auc": roc}, f, indent=2)
            mlflow.log_artifact("run_metrics.json")

            with open("classification_report.txt", "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact("classification_report.txt")

            log_confusion_matrix_plot(y_test, y_pred, "nn_confusion.png")
            log_prob_hist_plot(y_proba, "nn_prob_hist.png")

            if f1 > best_nn["f1"]:
                best_nn.update(
                    {"f1": f1, "run_id": run.info.run_id, "pipe": pipe}
                )

    # Permutation importance for best NN
    if best_nn["run_id"]:
        names = feat_names_from_preprocessor(prep, num_cols, cat_cols)
        with mlflow.start_run(run_id=best_nn["run_id"]):
            top, val = perm_top_for_best(
                best_nn["pipe"], X_test, y_test, names, "nn"
            )
            mlflow.log_param("most_informative_feature", top)
            mlflow.log_metric("most_informative_importance", val)

    return best_nn


def run_svc(X_train, X_test, y_train, y_test, prep, num_cols, cat_cols):
    svc_grid = [
        {"kernel": "linear", "C": 1, "gamma": "scale"},
        {"kernel": "linear", "C": 10, "gamma": "scale"},
        {"kernel": "rbf", "C": 1, "gamma": "scale"},
        {"kernel": "rbf", "C": 10, "gamma": "scale"},
    ]

    best_svc = {"f1": -np.inf, "run_id": None, "pipe": None}

    for p in svc_grid:
        name = f"SVC_k{p['kernel']}_C{p['C']}_g{p['gamma']}"
        with mlflow.start_run(run_name=name) as run:
            model = SVC(
                kernel=p["kernel"],
                C=p["C"],
                gamma=p["gamma"],
                probability=True,
                cache_size=500,
            )
            pipe = Pipeline([("prep", prep), ("clf", model)])
            pipe.fit(X_train, y_train)

            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred))
            roc = float(roc_auc_score(y_test, y_proba))

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc)

            mlflow.set_tag("primary_metric", "f1")
            mlflow.set_tag("primary_metric_goal", "maximize")
            mlflow.set_tag("model_family", "SVC")

            with open("run_metrics.json", "w") as f:
                json.dump({"accuracy": acc, "f1": f1, "roc_auc": roc}, f, indent=2)
            mlflow.log_artifact("run_metrics.json")

            with open("classification_report.txt", "w") as f:
                f.write(classification_report(y_test, y_pred))
            mlflow.log_artifact("classification_report.txt")

            log_confusion_matrix_plot(y_test, y_pred, "svc_confusion.png")
            log_prob_hist_plot(y_proba, "svc_prob_hist.png")

            if f1 > best_svc["f1"]:
                best_svc.update(
                    {"f1": f1, "run_id": run.info.run_id, "pipe": pipe}
                )

    # Permutation importance for best SVC
    if best_svc["run_id"]:
        names = feat_names_from_preprocessor(prep, num_cols, cat_cols)
        with mlflow.start_run(run_id=best_svc["run_id"]):
            top, val = perm_top_for_best(
                best_svc["pipe"], X_test, y_test, names, "svc"
            )
            mlflow.log_param("most_informative_feature", top)
            mlflow.log_metric("most_informative_importance", val)

    return best_svc


def print_experiment_link(experiment_name: str):
    from urllib.parse import quote

    host = (os.environ.get("DATABRICKS_HOST") or "").rstrip("/")
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp and host:
        exp_url = f"{host}/#mlflow/experiments/{quote(exp.experiment_id)}"
        print("🔗 Open your experiment:", exp_url)

    print("\n🧭 In the Databricks Experiments UI:")
    print("   • Click 'Add column' → metrics: accuracy, f1, roc_auc")
    print("   • Sort by f1 (descending) to see the best runs")
    print("   • You can also filter by tag 'model_family' (MLPClassifier / SVC)")
    print("\n✅ Hands-On 3 (Classification): all runs logged.")


def main():
    experiment_name = setup_mlflow()

    df = robust_read_csv_local(CSV_PATH)
    df = engineer_target(df)
    X, y, prep, num_cols, cat_cols = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_nn = run_nn(X_train, X_test, y_train, y_test, prep, num_cols, cat_cols)
    best_svc = run_svc(X_train, X_test, y_train, y_test, prep, num_cols, cat_cols)

    print("Best NN F1:", best_nn["f1"])
    print("Best SVC F1:", best_svc["f1"])

    print_experiment_link(f"/Users/{USER_EMAIL}/{EXP_NAME}")


if __name__ == "__main__":
    main()
