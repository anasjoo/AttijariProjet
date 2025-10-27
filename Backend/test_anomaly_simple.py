"""
Test simple de détection d'anomalies avec vos données existantes
"""
import pandas as pd
import numpy as np
from anomaly_detector_complete import BankingAnomalyDetector, test_anomaly_detection

def test_with_real_data():
    """Test avec vos données clients et transactions"""
    
    # Charger les données
    try:
        clients_df = pd.read_csv('datasets/fake_clients.csv')
        transactions_df = pd.read_csv('datasets/fake_transactions.csv')
        
        print("✅ Données chargées avec succès")
        print(f"Clients: {len(clients_df)} lignes")
        print(f"Transactions: {len(transactions_df)} lignes")
        
        # Préparer les données pour la détection
        # Fusionner pour avoir les infos clients avec les transactions
        full_df = transactions_df.merge(clients_df, on='client_id', how='left')
        
        # Tester la détection
        detector = BankingAnomalyDetector()
        anomalies = detector.detect_all_anomalies(full_df)
        
        if not anomalies.empty:
            print(f"\n🔍 {len(anomalies)} anomalies détectées")
            print("\nExemples d'anomalies:")
            print(anomalies[['client_id', 'montant', 'type_transaction', 'date']].head())
            
            # Résumé
            summary = detector.get_anomaly_summary(anomalies)
            print("\n📊 Résumé des anomalies:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
                
        else:
            print("✅ Aucune anomalie détectée")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("Utilisation des données exemple...")
        test_anomaly_detection()

if __name__ == "__main__":
    print("🧪 Test de détection d'anomalies bancaires")
    print("=" * 50)
    test_with_real_data()
