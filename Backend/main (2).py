"""
API FastAPI de production pour la détection d'anomalies bancaires
avec intégration LLM
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import uvicorn
from datetime import datetime
from typing import Optional, List, Dict
import os

from anomaly_detector_complete import BankingAnomalyDetector
from llm_explainer import LLMExplainer

# Initialiser l'application FastAPI
app = FastAPI(
    title="Banking Anomaly Detection API",
    description="API de production pour la détection d'anomalies bancaires avec LLM",
    version="1.0.0"
)

# Initialiser les composants
detector = BankingAnomalyDetector()
llm_explainer = LLMExplainer()

# Charger les données au démarrage
clients_df = pd.read_csv('../datasets/fake_clients.csv')
transactions_df = pd.read_csv('../datasets/fake_transactions.csv')
full_df = transactions_df.merge(clients_df, on='client_id', how='left')

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "Banking Anomaly Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(full_df)
    }

@app.get("/detect")
async def detect_anomalies(
    client_id: Optional[str] = None,
    limit: int = 100
):
    """Détecter les anomalies"""
    try:
        # Filtrer par client si spécifié
        df_to_analyze = full_df
        if client_id:
            df_to_analyze = full_df[full_df['client_id'] == client_id]
        
        # Détecter les anomalies
        anomalies = detector.detect_all_anomalies(df_to_analyze)
        
        if anomalies.empty:
            return {
                "anomalies": [],
                "summary": {"total_anomalies": 0},
                "message": "Aucune anomalie détectée"
            }
        
        # Limiter les résultats
        anomalies = anomalies.head(limit)
        
        # Convertir en dictionnaire
        anomalies_dict = anomalies.to_dict('records')
        
        # Générer le résumé
        summary = detector.get_anomaly_summary(anomalies)
        
        return {
            "anomalies": anomalies_dict,
            "summary": summary,
            "total_analyzed": len(df_to_analyze)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/anomalies/{client_id}")
async def get_client_anomalies(client_id: str):
    """Obtenir les anomalies pour un client spécifique"""
    try:
        client_data = full_df[full_df['client_id'] == client_id]
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client non trouvé")
        
        anomalies = detector.detect_all_anomalies(client_data)
        
        if anomalies.empty:
            return {
                "client_id": client_id,
                "anomalies": [],
                "message": "Aucune anomalie détectée pour ce client"
            }
        
        return {
            "client_id": client_id,
            "anomalies": anomalies.to_dict('records'),
            "summary": detector.get_anomaly_summary(anomalies)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{client_id}")
async def explain_anomalies(client_id: str):
    """Obtenir une explication LLM des anomalies pour un client"""
    try:
        client_data = full_df[full_df['client_id'] == client_id]
        
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client non trouvé")
        
        anomalies = detector.detect_all_anomalies(client_data)
        
        if anomalies.empty:
            return {
                "client_id": client_id,
                "explanation": "Aucune anomalie détectée",
                "recommendations": []
            }
        
        # Obtenir l'explication LLM
        explanation = llm_explainer.explain_anomalies(
            client_data.iloc[0], 
            anomalies.iloc[0] if not anomalies.empty else None
        )
        
        return {
            "client_id": client_id,
            "anomalies_count": len(anomalies),
            "explanation": explanation,
            "anomalies": anomalies.to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def generate_report():
    """Générer un rapport complet"""
    try:
        anomalies = detector.detect_all_anomalies(full_df)
        summary = detector.get_anomaly_summary(anomalies)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_transactions": len(full_df),
            "total_anomalies": summary["total_anomalies"],
            "anomaly_rate": summary["total_anomalies"] / len(full_df) * 100,
            "summary": summary,
            "top_risk_clients": anomalies['client_id'].value_counts().head(5).to_dict() if not anomalies.empty else {}
        }
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
