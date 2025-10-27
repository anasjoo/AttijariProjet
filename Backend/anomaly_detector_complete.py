import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class BankingAnomalyDetector:
    """
    Détection complète d'anomalies bancaires avec toutes les méthodes
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
    def detect_all_anomalies(self, df_transactions):
        """
        Détection complète avec toutes les méthodes
        """
        df = df_transactions.copy()
        
        # 1. Z-score pour valeurs extrêmes
        df = self._detect_zscore_anomalies(df)
        
        # 2. Isolation Forest
        df = self._detect_isolation_forest(df)
        
        # 3. Règles métier bancaires
        df = self._detect_banking_rules(df)
        
        # 4. Combinaison finale
        df['anomaly_detected'] = (
            df['anomaly_zscore'] | 
            df['anomaly_isolation'] | 
            df['anomaly_high_amount'] |
            df['anomaly_unusual_country'] |
            df['anomaly_unknown_recipient']
        )
        
        return df[df['anomaly_detected'] == True]
    
    def _detect_zscore_anomalies(self, df):
        """Z-score pour détecter valeurs extrêmes"""
        stats = df.groupby("client_id")["montant"].agg(["mean", "std"]).reset_index()
        stats.columns = ["client_id", "montant_mean", "montant_std"]
        
        df = df.merge(stats, on="client_id", how="left")
        df["z_score"] = (df["montant"] - df["montant_mean"]) / df["montant_std"]
        df["anomaly_zscore"] = df["z_score"].abs() > 3
        
        return df
    
    def _detect_isolation_forest(self, df):
        """Isolation Forest pour anomalies multivariées"""
        features = ['montant', 'nb_transactions_7j', 'montant_moyen_7j']
        
        if all(f in df.columns for f in features):
            X = df[features].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            anomalies = self.isolation_forest.fit_predict(X_scaled)
            df['anomaly_isolation'] = anomalies == -1
        else:
            df['anomaly_isolation'] = False
            
        return df
    
    def _detect_banking_rules(self, df):
        """Règles métier spécifiques bancaires"""
        
        # 1. Montant élevé
        client_stats = df.groupby("client_id")["montant"].agg(["mean", "std"]).reset_index()
        df = df.merge(client_stats, on="client_id", how="left", suffixes=('', '_client'))
        
        # Pour les petits échantillons, utiliser un seuil absolu plus bas
        df['anomaly_high_amount'] = df.apply(
            lambda row: row['montant'] > max(row['montant_mean'] + 2 * row['montant_std'], 5000) 
            if row['montant_std'] > 0 else row['montant'] > 5000,
            axis=1
        )
        
        # 2. Pays inhabituel
        if 'pays' in df.columns:
            client_countries = df.groupby("client_id")["pays"].apply(set).reset_index()
            client_countries.columns = ["client_id", "pays_habituels"]
            df = df.merge(client_countries, on="client_id", how="left")
            df['anomaly_unusual_country'] = df.apply(
                lambda row: row['pays'] not in row['pays_habituels'] 
                if pd.notna(row['pays']) else False, 
                axis=1
            )
        else:
            df['anomaly_unusual_country'] = False
        
        # 3. Destinataire inconnu
        if 'destination' in df.columns:
            client_recipients = df.groupby("client_id")["destination"].apply(set).reset_index()
            client_recipients.columns = ["client_id", "destinataires_connus"]
            df = df.merge(client_recipients, on="client_id", how="left")
            df['anomaly_unknown_recipient'] = df.apply(
                lambda row: (
                    row['destination'] not in row['destinataires_connus'] and
                    row['montant'] > row['montant_mean'] * 2
                ) if pd.notna(row['destination']) else False,
                axis=1
            )
        else:
            df['anomaly_unknown_recipient'] = False
            
        return df
    
    def get_anomaly_summary(self, anomalies_df):
        """Résumé des anomalies détectées"""
        if anomalies_df.empty:
            return {"total_anomalies": 0}

        # Convert NumPy scalars to native Python types
        summary = {
            "total_anomalies": int(len(anomalies_df)),
            "zscore_anomalies": int(anomalies_df['anomaly_zscore'].sum()),
            "isolation_anomalies": int(anomalies_df['anomaly_isolation'].sum()),
            "high_amount_anomalies": int(anomalies_df['anomaly_high_amount'].sum()),
            "unusual_country_anomalies": int(anomalies_df['anomaly_unusual_country'].sum()),
            "unknown_recipient_anomalies": int(anomalies_df['anomaly_unknown_recipient'].sum())
        }
        return summary

# Fonction utilitaire pour test rapide
def test_anomaly_detection():
    """Test rapide avec données exemple"""
    import pandas as pd
    
    # Données exemple
    data = {
        'client_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C003'],
        'montant': [100, 150, 5000, 200, 50, 10000],
        'pays': ['TN', 'TN', 'FR', 'TN', 'TN', 'US'],
        'destination': ['D1', 'D1', 'D2', 'D3', 'D3', 'D4'],
        'date': pd.date_range('2024-01-01', periods=6),
        'nb_transactions_7j': [1, 2, 1, 1, 1, 1],
        'montant_moyen_7j': [100, 125, 125, 200, 125, 10000]
    }
    
    df = pd.DataFrame(data)
    detector = BankingAnomalyDetector()
    anomalies = detector.detect_all_anomalies(df)
    
    print("Anomalies détectées:")
    print(anomalies[['client_id', 'montant', 'pays', 'destination']])
    
    return anomalies

if __name__ == "__main__":
    test_anomaly_detection()
