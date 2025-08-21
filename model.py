import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
from google.cloud import storage 

RANDOM_STATE = 42
DATA_BUCKET = os.environ.get("BUCKET")
DATASET = os.environ.get("OUTFILE")  
DATA_PATH = os.environ.get("DATA_PATH", "dataset.csv") 
MODEL_PATH = "model.joblib"
MODEL_UPLOAD_PATH = "models/model.joblib"
MODEL_BUCKET = "finure-models"
MODEL_GCS_URI = f"gs://{MODEL_BUCKET}/{MODEL_UPLOAD_PATH}"

if DATA_BUCKET and DATASET:
    uri = f"gs://{DATA_BUCKET}/{DATASET}"
    print(f"Downloading dataset from {uri} and saving as {DATA_PATH}")
    client = storage.Client()
    bucket = client.bucket(DATA_BUCKET)
    blob = bucket.blob(DATASET.lstrip("/"))
    os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
    blob.download_to_filename(DATA_PATH)
    
# Sample for final prediction
SAMPLE_INPUT: Dict[str, Any] = {
  "Age": 22,
  "Income": 97536,
  "Employed": 1,
  "CreditScore": 713,
  "LoanAmount": 784
}

def guess_target(df: pd.DataFrame) -> str:
    candidates = ["approved","approval","approval_status","card_approved","is_approved","target","label","class","y"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return df.columns[-1]

def stratified_split(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(X, y):
        return tr, te
    raise RuntimeError("Could not produce a stratified split.")

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in bool_cols:
        if c not in cat_cols:
            cat_cols.append(c)
        if c in numeric_cols:
            numeric_cols.remove(c)
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num_t, numeric_cols), ("cat", cat_t, cat_cols)], remainder="drop", verbose_feature_names_out=False)

def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    return make_pipeline(preprocessor, clf)

def print_summary(df: pd.DataFrame, target_col: str) -> None:
    print("\n DATASET SUMMARY ")
    print(f"Path: {DATA_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns and dtypes:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head(5).to_string())
    print(f"\nGuessed target column: {target_col!r}")
    print("\nTarget value counts:")
    print(df[target_col].value_counts(dropna=False))
    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

def evaluate(y_true, y_pred, y_prob: Optional[np.ndarray]):
    print("\n TEST SET EVAL ")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    average = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    try:
        print(f"Precision: {precision_score(y_true, y_pred, average=average, zero_division=0):.4f}")
        print(f"Recall   : {recall_score(y_true, y_pred, average=average, zero_division=0):.4f}")
        print(f"F1-score : {f1_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    except Exception as e:
        print(f"Could not compute PRF1: {e}")
    try:
        if y_prob is not None:
            if y_prob.ndim == 1:
                auc = roc_auc_score(y_true, y_prob)
            else:
                if y_prob.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
            print(f"ROC-AUC  : {auc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC-AUC: {e}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    try:
        print(classification_report(y_true, y_pred, zero_division=0))
    except Exception as e:
        print(f"Could not produce classification report: {e}")

def main() -> int:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}", file=sys.stderr)
        return 1
    df = pd.read_csv(DATA_PATH)
    target = guess_target(df)
    print_summary(df, target)

    y = df[target]
    X = df.drop(columns=[target])

    # Normalize common string targets to binary
    if y.dtype == object:
        yl = y.astype(str).str.strip().str.lower()
        mapping = {"yes":1,"y":1,"true":1,"t":1,"1":1,"approved":1,"approve":1,"positive":1,
                    "no":0,"n":0,"false":0,"f":0,"0":0,"rejected":0,"declined":0,"negative":0}
        if set(yl.unique()).issubset(set(mapping.keys())):
            y = yl.map(mapping).astype(int)

    tr_idx, te_idx = stratified_split(X, y, n_splits=5)
    X_train, X_test = X.iloc[tr_idx].copy(), X.iloc[te_idx].copy()
    y_train, y_test = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()

    pre = build_preprocessor(X_train)
    model = build_model(pre)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None
    evaluate(y_test, y_pred, y_prob)

    import joblib
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    try:
        client = storage.Client()  
        bucket = client.bucket(MODEL_BUCKET)
        blob = bucket.blob(MODEL_UPLOAD_PATH)
        blob.upload_from_filename(MODEL_PATH, content_type="application/octet-stream")
        print(f"Model uploaded to: {MODEL_GCS_URI}")
    except Exception as e:
        print(f"ERROR: Failed to upload model to GCS: {e}", file=sys.stderr)
        return 1

    # Test static sample
    sample_df = pd.DataFrame([SAMPLE_INPUT], columns=X_train.columns)
    pred = model.predict(sample_df)[0]
    ppos = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(sample_df)
            if prob.ndim == 2 and prob.shape[1] >= 2:
                ppos = prob[0,1]
            else:
                ppos = float(prob.ravel()[0])
        except Exception:
            ppos = None

    print("\n Prediction for hardcoded sample ")
    print("Input:")
    print(json.dumps(SAMPLE_INPUT, indent=2, default=str))
    print(f"Predicted class: {pred}")
    if ppos is not None:
        print(f"Estimated positive-class probability: {ppos:.4f}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
