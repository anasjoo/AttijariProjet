"""
Script de production pour la détection d'anomalies bancaires
Usage: python production_main.py
"""
import pandas as pd
import numpy as np
from datetime import datetime
from anomaly_detector_complete import BankingAnomalyDetector

def main():
    """Fonction principale pour la détection d'anomalies en production"""
    
    print("🚀 Démarrage de la détection d'anomalies en production...")
    
    # 1. Charger les données
    clients_df = pd.read_csv('../datasets/fake_clients.csv')
    transactions_df = pd.read_csv('../datasets/fake_transactions.csv')
    
    print(f"📊 Données chargées: {len(transactions_df)} transactions, {len(clients_df)} clients")
    
    # 2. Fusionner les données
    full_df = transactions_df.merge(clients_df, on='client_id', how='left')
    
    # 3. Détecter les anomalies
    detector = BankingAnomalyDetector()
    anomalies = detector.detect_all_anomalies(full_df)
    
    # 4. Générer le rapport
    summary = detector.get_anomaly_summary(anomalies)
    
    # 5. Afficher les résultats
    print("\n" + "="*60)
    print("📊 RAPPORT DE PRODUCTION")
    print("="*60)
    print(f"Transactions analysées: {len(full_df)}")
    print(f"Anomalies détectées: {summary['total_anomalies']}")
    print(f"Taux d'anomalie: {summary['total_anomalies']/len(full_df)*100:.2f}%")
    
    if not anomalies.empty:
        print("\n🚨 Top 10 anomalies:")
        print(anomalies[['client_id', 'montant', 'date']].head(10))
        
        print("\n📈 Résumé par type:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # 6. Sauvegarder les résultats
    if not anomalies.empty:
        anomalies.to_csv(f'anomalies_{datetime.now().strftime("%Y%m%d")}.csv', index=False)
        print(f"\n✅ Résultats sauvegardés: anomalies_{datetime.now().strftime('%Y%m%d')}.csv")
    
    return anomalies, summary

if __name__ == "__main__":
    anomalies, summary = main()
