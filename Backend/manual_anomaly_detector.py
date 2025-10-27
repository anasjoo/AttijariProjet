"""
Manual Anomaly Detection System - Fixed Version
Allows real-time anomaly detection when entering new client data manually
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ManualAnomalyDetector:
    """
    System for manual data entry with real-time anomaly detection
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_trained = False
        self.model_path = 'Backend/models/manual_anomaly_model.pkl'
        self.scaler_path = 'Backend/models/manual_anomaly_scaler.pkl'
        
    def ensure_model_trained(self):
        """Ensure model is trained and ready for use"""
        try:
            if not os.path.exists(self.model_path):
                print("🔄 Training new model for manual anomaly detection...")
                self.train_model()
            else:
                self.load_model()
        except Exception as e:
            print(f"❌ Error ensuring model exists: {e}")
            self.train_model()
    
    def train_model(self):
        """Train the anomaly detection model"""
        try:
            # Create synthetic training data
            np.random.seed(42)
            n_samples = 1000
            
            # Normal transaction patterns
            normal_data = {
                'montant': np.random.normal(1000, 300, 800),
                'revenu_mensuel': np.random.normal(3000, 1000, 800),
                'age': np.random.randint(25, 65, 800),
                'nb_transactions_7j': np.random.poisson(3, 800)
            }
            
            # Anomalous patterns
            anomaly_data = {
                'montant': np.random.normal(5000, 2000, 200),
                'revenu_mensuel': np.random.normal(3000, 1000, 200),
                'age': np.random.randint(25, 65, 200),
                'nb_transactions_7j': np.random.poisson(10, 200)
            }
            
            # Combine data
            df = pd.DataFrame({
                **{k: np.concatenate([normal_data[k], anomaly_data[k]]) 
                   for k in normal_data.keys()},
                'is_anomaly': [0] * 800 + [1] * 200
            })
            
            # Create features
            df['ratio_montant_revenu'] = df['montant'] / df['revenu_mensuel']
            df['z_score_montant'] = np.abs((df['montant'] - df['montant'].mean()) / df['montant'].std())
            
            features = ['montant', 'revenu_mensuel', 'age', 'nb_transactions_7j', 
                       'ratio_montant_revenu', 'z_score_montant']
            
            X = df[features].values
            y = df['is_anomaly'].values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = IsolationForest(
                contamination=0.2,
                random_state=42,
                n_estimators=100
            )
            self.model.fit(X_scaled)
            
            # Save model
            os.makedirs('Backend/models', exist_ok=True)
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            self.model_trained = True
            print("✅ Model trained successfully")
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            raise
    
    def load_model(self):
        """Load existing trained model"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.model_trained = True
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.train_model()
    
    def detect_manual_anomaly(self, client_data):
        """Detect anomaly in manually entered transaction data"""
        try:
            # Ensure model is ready
            self.ensure_model_trained()
            
            if self.model is None or self.scaler is None:
                raise ValueError("Model not properly initialized")
            
            # Prepare features from manual entry
            features = ['montant', 'revenu_mensuel', 'age', 'nb_transactions_7j', 
                       'ratio_montant_revenu', 'z_score_montant']
            
            # Create feature vector
            feature_vector = {
                'montant': float(client_data.get('montant', 0)),
                'revenu_mensuel': float(client_data.get('revenu_mensuel', 3000)),
                'age': int(client_data.get('age', 30)),
                'nb_transactions_7j': int(client_data.get('nb_transactions_7j', 3)),
                'ratio_montant_revenu': float(client_data.get('montant', 0)) / max(float(client_data.get('revenu_mensuel', 1)), 1),
                'z_score_montant': abs((float(client_data.get('montant', 0)) - 1500) / 1000)
            }
            
            # Create DataFrame
            X = pd.DataFrame([feature_vector])[features].values
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            anomaly_score = self.model.decision_function(X_scaled)[0]
            is_anomaly = self.model.predict(X_scaled)[0] == -1
            
            # Calculate confidence
            confidence = min(abs(anomaly_score) / 2, 1.0)
            
            return {
                'client_id': client_data.get('client_id', 'N/A'),
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'message': "Anomalie détectée" if is_anomaly else "Transaction normale"
            }
            
        except Exception as e:
            return {
                'client_id': client_data.get('client_id', 'N/A'),
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'message': f"Erreur: {str(e)}"
            }

# Global instance for easy access
manual_detector = ManualAnomalyDetector()
