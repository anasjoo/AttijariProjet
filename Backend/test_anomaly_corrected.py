"""
Test de détection d'anomalies avec chargement correct des fichiers CSV
"""
import pandas as pd
import numpy as np
import os
from anomaly_detector_complete import BankingAnomalyDetector

def test_with_real_datasets():
    """Test avec les vrais fichiers CSV depuis le dossier datasets"""
    
    try:
        # Chemins corrects vers les fichiers
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        clients_path = os.path.join(base_path, 'datasets', 'fake_clients.csv')
        transactions_path = os.path.join(base_path, 'datasets', 'fake_transactions.csv')
        
        print("📁 Chargement des fichiers...")
        print(f"Clients: {clients_path}")
        print(f"Transactions: {transactions_path}")
        
        # Charger les données
        clients_df = pd.read_csv(clients_path)
        transactions_df = pd.read_csv(transactions_path)
        
        print(f"\n✅ Données chargées avec succès!")
        print(f"   - Clients: {len(clients_df)} lignes")
        print(f"   - Transactions: {len(transactions_df)} lignes")
        
        # Afficher les colonnes disponibles
        print(f"\n📊 Colonnes disponibles:")
        print(f"   Clients: {list(clients_df.columns)}")
        print(f"   Transactions: {list(transactions_df.columns)}")
        
        # Préparer les données pour la détection
        full_df = transactions_df.merge(clients_df, on='client_id', how='left')
        print(f"\n🔗 Données fusionnées: {len(full_df)} lignes")
        
        # Tester la détection d'anomalies
        detector = BankingAnomalyDetector()
        anomalies = detector.detect_all_anomalies(full_df)
        
        if not anomalies.empty:
            print(f"\n🚨 {len(anomalies)} anomalies détectées!")
            
            # Afficher les détails
            print("\n📋 Détails des anomalies:")
            cols_to_show = ['client_id', 'montant', 'type_transaction', 'date', 'pays']
            available_cols = [col for col in cols_to_show if col in anomalies.columns]
            print(anomalies[available_cols].head(10))
            
            # Résumé par type
            summary = detector.get_anomaly_summary(anomalies)
            print("\n📈 Résumé par type d'anomalie:")
            for key, value in summary.items():
                print(f"   {key}: {value}")
                
            # Statistiques par client
            if 'client_id' in anomalies.columns:
                print("\n👥 Clients avec anomalies:")
                client_anomalies = anomalies['client_id'].value_counts().head(5)
                print(client_anomalies)
                
        else:
            print("✅ Aucune anomalie détectée dans les données")
            
        return anomalies
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        print("Utilisation des données exemple...")
        return test_with_example_data()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_with_example_data():
    """Test avec les données exemple intégrées"""
    from anomaly_detector_complete import test_anomaly_detection
    return test_anomaly_detection()

if __name__ == "__main__":
    print("🧪 Test de détection d'anomalies bancaires")
    print("=" * 60)
    anomalies = test_with_real_datasets()
    
    if anomalies is not None:
        print("\n✅ Test terminé avec succès!")
    else:
        print("\n❌ Test échoué")
