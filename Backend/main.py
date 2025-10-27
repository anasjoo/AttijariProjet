import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import List, Optional
import uuid
import pandas as pd

from hybrid_aml import feature_engineer, build_rules, ensemble_score, build_preprocessor

# Simple settings
API_TITLE = "Attijari Anomaly Detection API"
API_VERSION = "0.1.0"

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Pydantic models
class Txn(BaseModel):
    Transaction_ID: str
    Transaction_Amount: float
    Transaction_Volume: int
    Average_Transaction_Amount: float
    Frequency_of_Transactions: int
    Time_Since_Last_Transaction: int
    Day_of_Week: str
    Time_of_Day: str
    Age: int
    Gender: str
    Income: float
    Account_Type: str

# Load artifacts on startup
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
prep = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
iso = joblib.load(os.path.join(models_dir, "iso_clf.joblib"))
oc = joblib.load(os.path.join(models_dir, "ocsvm_clf.joblib"))
with open(os.path.join(models_dir, "schema.json")) as f:
    schema = json.load(f)
with open(os.path.join(models_dir, "rules.json")) as f:
    rules_cfg = json.load(f)

@app.post("/score-transaction")
def score_transaction(t: Txn):
    df = pd.DataFrame([t.dict()])
    df_fe = feature_engineer(df)

    # rebuild rules dataframe for this row using training thresholds from rules_cfg
    df_fe["Is_Weekend"] = df["Day_of_Week"].isin(["Saturday","Sunday"]).astype(int)
    rules = pd.DataFrame(index=df_fe.index)
    rules["R_amount_gt_iqr3x"] = (df_fe["Transaction_Amount"] > rules_cfg["amount_iqr3x"]).astype(int)
    rules["R_amt_gt_2x_avg_personal"] = (df_fe["Transaction_Amount"] > 2.0*df_fe["Average_Transaction_Amount"]).astype(int)
    rules["R_amt_lt_0_5x_avg_personal"] = (df_fe["Transaction_Amount"] < 0.5*df_fe["Average_Transaction_Amount"]).astype(int)
    rules["R_high_freq"] = (df_fe["Frequency_of_Transactions"] > 1.75*rules_cfg["freq_median"]).astype(int)
    rules["R_high_volume"] = (df_fe["Transaction_Volume"] >= rules_cfg["vol_q95"]).astype(int)
    rules["R_short_gap"] = (df_fe["Time_Since_Last_Transaction"] <= 2).astype(int)
    rules["R_young_high_amt"] = ((df_fe["Age"] < 21) & (df_fe["Transaction_Amount"] > rules_cfg["amount_q3"])).astype(int)
    rules["R_amt_gt_20pct_income"] = (df_fe["Transaction_Amount"] / max(df_fe["Income"].iloc[0],1) > (0.20/12)).astype(int)
    rules["R_savings_big"] = ((df_fe["Account_Type"]=="Savings") & (df_fe["Transaction_Amount"] > rules_cfg["amount_q3"])).astype(int)
    rules["R_night_activity"] = df_fe["Hour"].between(0,5).astype(int)
    rules["R_weekend_high"] = ((df_fe["Is_Weekend"]==1) & (df_fe["Transaction_Amount"] > rules_cfg["amount_q3"])).astype(int)
    rules["Rule_Score"] = rules.sum(axis=1)

    # models
    X = df_fe[schema["num_features"] + schema["cat_features"]]
    Z = prep.transform(X)
    # IF score (inverse + min-max single point safe fallback)
    if_score = -iso["model"].score_samples(Z)
    if_norm  = (if_score - if_score.min())/(if_score.max()-if_score.min()+1e-9)
    # OCSVM score
    oc_score = -oc["model"].decision_function(Z)
    oc_norm  = (oc_score - oc_score.min())/(oc_score.max()-oc_score.min()+1e-9)

    # ensemble
    rule_norm = (rules["Rule_Score"] - rules["Rule_Score"].min())/(rules["Rule_Score"].max()-rules["Rule_Score"].min()+1e-9)
    ensemble = 0.5*if_norm + 0.3*oc_norm + 0.2*rule_norm

    # threshold for real-time: fixed (e.g. 0.9) or business logic
    final_flag = int(ensemble.iloc[0] >= 0.9)

    return {
        "transaction_id": t.Transaction_ID,
        "ensemble_score": float(ensemble.iloc[0]),
        "if_score": float(if_norm.iloc[0]),
        "ocsvm_score": float(oc_norm.iloc[0]),
        "rule_score": int(rules["Rule_Score"].iloc[0]),
        "rule_hits": [c for c in rules.columns if c.startswith("R_") and int(rules[c].iloc[0])==1],
        "final_flag": final_flag,
        "model_version": "2025-09-13_aml_v1"
    }

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    Transaction_Amount: float
    Average_Transaction_Amount: float
    Frequency_of_Transactions: float
    model: str | None = None  # "autoencoder" or "isolation_forest"; None = auto-pick


class PredictResponse(BaseModel):
    is_anomaly: bool
    score: float
    threshold: float | None = None
    model: str
    explanation: str | None = None


class AnomalyRecord(BaseModel):
    id: str
    transaction_amount: float
    average_transaction_amount: float
    frequency_of_transactions: float
    date: str
    is_anomaly: bool
    score: float
    explanation: str
    action: str


class AnomalyListResponse(BaseModel):
    anomalies: List[AnomalyRecord]
    total_count: int
    anomaly_count: int


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Lazy globals for artifacts
_scaler = None
_ae_threshold = None
_ae_model = None
_if_model = None

# In-memory storage for demo purposes (in production, use a database)
_anomaly_storage = [
    {
        "id": "TXN-001",
        "transaction_amount": 15000.0,
        "average_transaction_amount": 5000.0,
        "frequency_of_transactions": 2.0,
        "date": "2024-01-15T10:00:00",
        "is_anomaly": True,
        "score": 0.85,
        "explanation": "Transaction amount significantly exceeds customer's average spending pattern.",
        "action": "Review Required",
        "transaction_volume": 1,
        "time_since_last_transaction": 24,
        "day_of_week": "Monday",
        "time_of_day": "Morning",
        "age": 35,
        "gender": "Male",
        "income": 60000.0,
        "account_type": "Savings"
    },
    {
        "id": "TXN-002",
        "transaction_amount": 250.0,
        "average_transaction_amount": 300.0,
        "frequency_of_transactions": 1.0,
        "date": "2024-01-15T14:00:00",
        "is_anomaly": False,
        "score": 0.12,
        "explanation": "Transaction falls within normal spending patterns.",
        "action": "Approved",
        "transaction_volume": 1,
        "time_since_last_transaction": 48,
        "day_of_week": "Monday",
        "time_of_day": "Afternoon",
        "age": 28,
        "gender": "Female",
        "income": 45000.0,
        "account_type": "Checking"
    },
    {
        "id": "TXN-003",
        "transaction_amount": 8500.0,
        "average_transaction_amount": 2000.0,
        "frequency_of_transactions": 5.0,
        "date": "2024-01-14T09:00:00",
        "is_anomaly": True,
        "score": 0.78,
        "explanation": "Unusual transaction frequency detected.",
        "action": "Flagged",
        "transaction_volume": 1,
        "time_since_last_transaction": 12,
        "day_of_week": "Sunday",
        "time_of_day": "Morning",
        "age": 42,
        "gender": "Male",
        "income": 80000.0,
        "account_type": "Savings"
    },
    {
        "id": "TXN-004",
        "transaction_amount": 1200.0,
        "average_transaction_amount": 1000.0,
        "frequency_of_transactions": 3.0,
        "date": "2024-01-14T16:00:00",
        "is_anomaly": False,
        "score": 0.23,
        "explanation": "Standard transaction within expected parameters.",
        "action": "Approved",
        "transaction_volume": 1,
        "time_since_last_transaction": 36,
        "day_of_week": "Sunday",
        "time_of_day": "Afternoon",
        "age": 31,
        "gender": "Female",
        "income": 55000.0,
        "account_type": "Checking"
    },
    {
        "id": "TXN-005",
        "transaction_amount": 25000.0,
        "average_transaction_amount": 6000.0,
        "frequency_of_transactions": 1.0,
        "date": "2024-01-13T22:00:00",
        "is_anomaly": True,
        "score": 0.92,
        "explanation": "High-value transaction outside normal business hours.",
        "action": "Manual Review",
        "transaction_volume": 1,
        "time_since_last_transaction": 72,
        "day_of_week": "Saturday",
        "time_of_day": "Night",
        "age": 55,
        "gender": "Male",
        "income": 120000.0,
        "account_type": "Business"
    }
]


def _is_reasonable_transaction(transaction_amount: float, average_amount: float, frequency: float) -> bool:
    """Business logic to determine if a transaction is reasonable regardless of ML model"""
    
    # Calculate ratio of transaction to average
    if average_amount > 0:
        ratio = transaction_amount / average_amount
    else:
        return True  # If no average, assume reasonable
    
    # Reasonable ranges based on business logic
    # Normal transactions: 0.1x to 5x the average
    if 0.1 <= ratio <= 5.0:
        return True
    
    # Very high amounts (>5x average) might be suspicious
    if ratio > 5.0:
        return False
    
    # Very low amounts (<0.1x average) might be suspicious
    if ratio < 0.1:
        return False
    
    return True


def _generate_llm_explanation(is_anomaly: bool, score: float, transaction_amount: float, 
                            average_amount: float, frequency: float) -> str:
    """Generate human-readable explanation for anomaly detection results"""
    
    if not is_anomaly:
        return "Transaction falls within normal spending patterns and frequency."
    
    explanations = []
    
    # Amount-based explanations
    amount_ratio = transaction_amount / average_amount if average_amount > 0 else 0
    
    if amount_ratio > 3:
        explanations.append(f"Transaction amount ({transaction_amount:,.0f} TND) is significantly higher than customer's average ({average_amount:,.0f} TND) - {amount_ratio:.1f}x higher")
    elif amount_ratio > 2:
        explanations.append(f"Transaction amount ({transaction_amount:,.0f} TND) exceeds customer's typical spending pattern by {amount_ratio:.1f}x")
    elif amount_ratio < 0.3:
        explanations.append(f"Transaction amount ({transaction_amount:,.0f} TND) is unusually low compared to customer's average ({average_amount:,.0f} TND)")
    
    # Frequency-based explanations
    if frequency > 10:
        explanations.append(f"Unusually high transaction frequency ({frequency:.1f} transactions per day)")
    elif frequency < 0.1:
        explanations.append(f"Unusually low transaction frequency ({frequency:.1f} transactions per day)")
    
    # Score-based explanations
    if score > 0.8:
        explanations.append("High-risk transaction pattern detected by machine learning model")
    elif score > 0.6:
        explanations.append("Moderate risk indicators present in transaction pattern")
    elif score > 0.4:
        explanations.append("Suspicious patterns detected in transaction data")
    
    if not explanations:
        explanations.append("Anomalous pattern detected by machine learning model")
    
    return ". ".join(explanations) + "."


def _get_recommended_action(is_anomaly: bool, score: float) -> str:
    """Get recommended action based on anomaly detection"""
    if not is_anomaly:
        return "Approved"
    
    if score > 0.8:
        return "Manual Review"
    elif score > 0.6:
        return "Review Required"
    else:
        return "Flagged"


def _load_optional_artifacts() -> None:
    global _scaler, _ae_threshold, _ae_model, _if_model
    # Load what exists; keep optional to let the skeleton run before training export
    try:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path) and _scaler is None:
            _scaler = joblib.load(scaler_path)
    except Exception:
        _scaler = None

    try:
        thr_path = os.path.join(MODELS_DIR, "ae_threshold.npy")
        if os.path.exists(thr_path) and _ae_threshold is None:
            _ae_threshold = float(np.load(thr_path))
    except Exception:
        _ae_threshold = None

    # Isolation Forest (optional)
    try:
        if_path = os.path.join(MODELS_DIR, "isolation_forest.pkl")
        if os.path.exists(if_path) and _if_model is None:
            _if_model = joblib.load(if_path)
    except Exception:
        _if_model = None

    # Autoencoder via joblib or keras SavedModel; we try joblib first
    try:
        ae_joblib = os.path.join(MODELS_DIR, "autoencoder.pkl")
        if os.path.exists(ae_joblib) and _ae_model is None:
            _ae_model = joblib.load(ae_joblib)
    except Exception:
        _ae_model = None

    # If keras format directory exists, attempt a delayed import to avoid TF dependency if unused
    if _ae_model is None:
        keras_dir = os.path.join(MODELS_DIR, "autoencoder_keras")
        if os.path.isdir(keras_dir):
            try:
                from tensorflow import keras  # type: ignore
                _ae_model = keras.models.load_model(keras_dir)
            except Exception:
                _ae_model = None


@app.get("/health")
def health():
    _load_optional_artifacts()
    return {
        "status": "ok",
        "has_scaler": _scaler is not None,
        "has_if": _if_model is not None,
        "has_ae": _ae_model is not None,
        "has_ae_threshold": _ae_threshold is not None,
        "total_anomalies_stored": len(_anomaly_storage),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    _load_optional_artifacts()

    features = np.array([[
        body.Transaction_Amount,
        body.Average_Transaction_Amount,
        body.Frequency_of_Transactions,
    ]])

    X = features
    if _scaler is not None:
        try:
            X = _scaler.transform(features)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Scaler failed: {exc}")

    # If caller explicitly asked for AE
    if (body.model == "autoencoder") and (_ae_model is not None and _ae_threshold is not None):
        try:
            recon = _ae_model.predict(X, verbose=0)
            mse = float(np.mean((X - recon) ** 2))
            is_anom = mse > float(_ae_threshold)
            explanation = _generate_llm_explanation(
                is_anom, mse, body.Transaction_Amount, 
                body.Average_Transaction_Amount, body.Frequency_of_Transactions
            )
            return PredictResponse(
                is_anomaly=bool(is_anom), 
                score=mse, 
                threshold=float(_ae_threshold), 
                model="autoencoder",
                explanation=explanation
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Autoencoder prediction failed: {exc}")

    # If caller explicitly asked for IF
    if (body.model == "isolation_forest") and (_if_model is not None):
        try:
            # Use decision scores when possible
            if hasattr(_if_model, "score_samples"):
                raw_score = float(_if_model.score_samples(X)[0])
                # Isolation Forest: lower scores = more anomalous
                # Convert to 0-1 scale where 1 = most anomalous
                score = max(0, 1 + raw_score)  # raw_score is typically negative for anomalies
                is_anom = raw_score < -0.3  # Much more conservative threshold
            else:
                pred = int(_if_model.predict(X)[0])
                score = 1.0 if pred == -1 else 0.0
                is_anom = pred == -1
            explanation = _generate_llm_explanation(
                is_anom, score, body.Transaction_Amount, 
                body.Average_Transaction_Amount, body.Frequency_of_Transactions
            )
            return PredictResponse(
                is_anomaly=bool(is_anom), 
                score=score, 
                threshold=-0.3, 
                model="isolation_forest",
                explanation=explanation
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Isolation Forest prediction failed: {exc}")

    # Auto-pick: prefer Isolation Forest (performed better), else Autoencoder
    if _if_model is not None:
        try:
            if hasattr(_if_model, "score_samples"):
                raw_score = float(_if_model.score_samples(X)[0])
                # Isolation Forest: lower scores = more anomalous
                # Convert to 0-1 scale where 1 = most anomalous
                score = max(0, 1 + raw_score)  # raw_score is typically negative for anomalies
                ml_anomaly = raw_score < -0.3  # Much more conservative threshold
            else:
                pred = int(_if_model.predict(X)[0])
                score = 1.0 if pred == -1 else 0.0
                ml_anomaly = pred == -1
            
            # Apply business logic override
            is_reasonable = _is_reasonable_transaction(
                body.Transaction_Amount, 
                body.Average_Transaction_Amount, 
                body.Frequency_of_Transactions
            )
            
            # Final decision: ML anomaly AND not reasonable by business logic
            is_anom = ml_anomaly and not is_reasonable
            
            explanation = _generate_llm_explanation(
                is_anom, score, body.Transaction_Amount, 
                body.Average_Transaction_Amount, body.Frequency_of_Transactions
            )
            return PredictResponse(
                is_anomaly=bool(is_anom), 
                score=score, 
                threshold=-0.3, 
                model="isolation_forest",
                explanation=explanation
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Isolation Forest prediction failed: {exc}")

    if _ae_model is not None and _ae_threshold is not None:
        try:
            recon = _ae_model.predict(X, verbose=0)
            mse = float(np.mean((X - recon) ** 2))
            is_anom = mse > float(_ae_threshold)
            explanation = _generate_llm_explanation(
                is_anom, mse, body.Transaction_Amount, 
                body.Average_Transaction_Amount, body.Frequency_of_Transactions
            )
            return PredictResponse(
                is_anomaly=bool(is_anom), 
                score=mse, 
                threshold=float(_ae_threshold), 
                model="autoencoder",
                explanation=explanation
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Autoencoder prediction failed: {exc}")

    raise HTTPException(status_code=503, detail="No model artifacts available. Train/export models first.")


@app.get("/anomalies", response_model=AnomalyListResponse)
def get_anomalies(limit: int = 50, offset: int = 0):
    """Get stored anomaly records"""
    total_count = len(_anomaly_storage)
    anomaly_count = sum(1 for record in _anomaly_storage if record["is_anomaly"])
    
    # Apply pagination
    start_idx = offset
    end_idx = min(offset + limit, total_count)
    records = _anomaly_storage[start_idx:end_idx]
    
    return AnomalyListResponse(
        anomalies=[AnomalyRecord(**record) for record in records],
        total_count=total_count,
        anomaly_count=anomaly_count
    )


@app.post("/anomalies")
def store_anomaly(body: PredictRequest):
    """Store a prediction result as an anomaly record"""
    prediction = predict(body)
    
    record = {
        "id": str(uuid.uuid4()),
        "transaction_amount": body.Transaction_Amount,
        "average_transaction_amount": body.Average_Transaction_Amount,
        "frequency_of_transactions": body.Frequency_of_Transactions,
        "date": datetime.now().isoformat(),
        "is_anomaly": prediction.is_anomaly,
        "score": prediction.score,
        "explanation": prediction.explanation or "No explanation available",
        "action": _get_recommended_action(prediction.is_anomaly, prediction.score)
    }
    
    _anomaly_storage.append(record)
    
    return {"message": "Anomaly record stored", "record_id": record["id"]}


@app.get("/debug-predict")
def debug_predict(transaction_amount: float, average_amount: float, frequency: float):
    """Debug endpoint to understand model behavior"""
    _load_optional_artifacts()
    
    features = np.array([[transaction_amount, average_amount, frequency]])
    X = features
    if _scaler is not None:
        X = _scaler.transform(features)
    
    debug_info = {
        "input_features": [transaction_amount, average_amount, frequency],
        "scaled_features": X[0].tolist() if _scaler else "No scaler",
        "has_scaler": _scaler is not None,
        "has_if_model": _if_model is not None,
        "has_ae_model": _ae_model is not None,
    }
    
    if _if_model is not None:
        try:
            raw_score = float(_if_model.score_samples(X)[0])
            pred = int(_if_model.predict(X)[0])
            debug_info["isolation_forest"] = {
                "raw_score": raw_score,
                "prediction": pred,
                "is_anomaly_by_score": raw_score < -0.1,
                "is_anomaly_by_pred": pred == -1,
                "score_normalized": max(0, 1 + raw_score)
            }
        except Exception as e:
            debug_info["isolation_forest_error"] = str(e)
    
    return debug_info


@app.get("/stats")
def get_stats():
    """Get overall statistics"""
    total_count = len(_anomaly_storage)
    anomaly_count = sum(1 for record in _anomaly_storage if record["is_anomaly"])
    normal_count = total_count - anomaly_count
    
    if total_count > 0:
        accuracy_rate = (anomaly_count / total_count) * 100
        avg_score = sum(record["score"] for record in _anomaly_storage) / total_count
        risk_level = "High" if avg_score > 0.7 else "Medium" if avg_score > 0.4 else "Low"
    else:
        accuracy_rate = 0
        avg_score = 0
        risk_level = "Low"
    
    return {
        "total_transactions": total_count,
        "anomalies_detected": anomaly_count,
        "normal_transactions": normal_count,
        "accuracy_rate": round(accuracy_rate, 2),
        "average_risk_score": round(avg_score, 3),
        "overall_risk_level": risk_level
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


