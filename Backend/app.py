"""
API FastAPI de production pour la détection d'anomalies bancaires
avec intégration LLM
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import uvicorn
from datetime import datetime
from typing import Optional, List, Dict
import os

from anomaly_detector_complete import BankingAnomalyDetector
from banking_llm_simple import BankingLLM

# Initialiser l'application FastAPI
app = FastAPI(
    title="Banking Anomaly Detection API",
    description="API de production pour la détection d'anomalies bancaires avec LLM",
    version="1.0.0"
)

# Initialiser les composants
detector = BankingAnomalyDetector()
llm = BankingLLM()

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
        "status": "running",
        "data_loaded": len(full_df)
    }

@app.get("/health")
async def health_check():
    """Vérification de santé"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_transactions": len(full_df),
        "total_clients": len(clients_df)
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
        
        # Convertir en dictionnaire avec types natifs
        anomalies = anomalies.replace({np.nan: None})
        anomalies_dict = anomalies.astype(object).to_dict('records')
        
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
        
        # Obtenir les informations du client
        client_info = client_data.iloc[0]
        anomaly_info = anomalies.iloc[0]
        
        # Créer la transaction pour l'explication
        transaction_desc = f"Transaction de {anomaly_info['montant']} TND le {anomaly_info['date']}"
        
        # Obtenir l'explication LLM
        explanation = llm.expliquer_transaction(
            client_id=client_id,
            profession=client_info['profession'],
            revenu=client_info['revenu_mensuel'],
            transaction=transaction_desc
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
            "total_clients": len(clients_df),
            "total_anomalies": summary["total_anomalies"],
            "anomaly_rate": summary["total_anomalies"] / len(full_df) * 100,
            "summary": summary,
            "top_risk_clients": anomalies['client_id'].value_counts().head(5).to_dict() if not anomalies.empty else {}
        }
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-detect")
async def batch_detect(transactions: List[Dict]):
    """Détecter les anomalies sur un lot de transactions"""
    try:
        # Convertir en DataFrame
        batch_df = pd.DataFrame(transactions)
        
        # Détecter les anomalies
        anomalies = detector.detect_all_anomalies(batch_df)
        
        return {
            "total_processed": len(batch_df),
            "anomalies_found": len(anomalies),
            "anomalies": anomalies.to_dict('records') if not anomalies.empty else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
