"""
Pipeline CRISP-DM Complet pour la Détection d'Anomalies Bancaires
Orchestration des étapes: Data Understanding → Data Preparation → Modeling
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ajouter le répertoire Backend au path
sys.path.append(str(Path(__file__).parent))

from data_understanding import BankingDataExplorer
from data_preparation import BankingDataPreprocessor
from anomaly_detection_ml import BankingAnomalyDetector

class CRISPDMPipeline:
    """Pipeline complet CRISP-DM pour la détection d'anomalies bancaires"""
    
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.explorer = None
        self.preprocessor = None
        self.detector = None
        
        # Chemins des fichiers
        self.cleaned_data_path = Path("../datasets/cleaned_banking_data.csv")
        self.anomaly_results_path = Path("../datasets/anomaly_detection_results.csv")
        
    def run_data_understanding(self, sample_size=50000):
        """Phase 1: Compréhension des données (EDA)"""
        print("🚀" + "="*80)
        print("PHASE 1: COMPRÉHENSION DES DONNÉES (DATA UNDERSTANDING)")
        print("="*80)
        
        self.explorer = BankingDataExplorer(self.raw_data_path)
        df = self.explorer.load_data(sample_size=sample_size)
        
        # Exploration complète
        self.explorer.basic_info()
        self.explorer.explore_categorical_variables()
        self.explorer.explore_numerical_variables()
        self.explorer.explore_dates()
        self.explorer.explore_clients()
        self.explorer.explore_transactions()
        
        # Générer les visualisations
        self.explorer.generate_visualizations()
        
        # Rapport final
        self.explorer.generate_summary_report()
        
        print("\n✅ Phase 1 terminée: Compréhension des données")
        return df
    
    def run_data_preparation(self, sample_size=100000):
        """Phase 2: Préparation des données"""
        print("\n🚀" + "="*80)
        print("PHASE 2: PRÉPARATION DES DONNÉES (DATA PREPARATION)")
        print("="*80)
        
        self.preprocessor = BankingDataPreprocessor(self.raw_data_path)
        
        # Charger les données
        self.preprocessor.load_data(sample_size=sample_size)
        
        # Nettoyer les données
        self.preprocessor.clean_data()
        
        # Créer les features
        self.preprocessor.create_features()
        
        # Générer le rapport
        self.preprocessor.generate_summary_report()
        
        # Sauvegarder les données nettoyées
        self.preprocessor.save_cleaned_data(self.cleaned_data_path)
        
        print("\n✅ Phase 2 terminée: Préparation des données")
        return self.preprocessor.df_cleaned
    
    def run_modeling(self):
        """Phase 3: Modélisation (Détection d'anomalies)"""
        print("\n🚀" + "="*80)
        print("PHASE 3: MODÉLISATION (DÉTECTION D'ANOMALIES)")
        print("="*80)
        
        self.detector = BankingAnomalyDetector(self.cleaned_data_path)
        
        # Charger les données nettoyées
        self.detector.load_data()
        
        # Préparer les features pour le ML
        df_ml, features = self.detector.prepare_features_for_ml()
        
        # Entraîner les modèles
        print("\n🤖 Entraînement des modèles de détection d'anomalies:")
        
        # Isolation Forest
        self.detector.train_isolation_forest(contamination=0.1)
        
        # One-Class SVM
        self.detector.train_one_class_svm(nu=0.1)
        
        # DBSCAN
        self.detector.train_dbscan(eps=0.5, min_samples=5)
        
        # Analyser les anomalies
        results_df = self.detector.analyze_anomalies()
        
        # Visualiser les résultats
        self.detector.visualize_anomalies()
        
        # Générer le rapport
        self.detector.generate_anomaly_report()
        
        # Sauvegarder les résultats
        self.detector.save_results(self.anomaly_results_path)
        
        print("\n✅ Phase 3 terminée: Modélisation")
        return self.detector, results_df
    
    def run_complete_pipeline(self, exploration_sample=50000, preparation_sample=100000):
        """Exécuter le pipeline complet CRISP-DM"""
        print("🎯" + "="*80)
        print("PIPELINE CRISP-DM COMPLET - DÉTECTION D'ANOMALIES BANCAIRES")
        print("="*80)
        
        try:
            # Phase 1: Data Understanding
            df_exploration = self.run_data_understanding(exploration_sample)
            
            # Phase 2: Data Preparation
            df_cleaned = self.run_data_preparation(preparation_sample)
            
            # Phase 3: Modeling
            detector, results = self.run_modeling()
            
            # Résumé final
            self.generate_final_summary()
            
            print("\n🎉" + "="*80)
            print("PIPELINE CRISP-DM TERMINÉ AVEC SUCCÈS!")
            print("="*80)
            
            return {
                'explorer': self.explorer,
                'preprocessor': self.preprocessor,
                'detector': detector,
                'results': results
            }
            
        except Exception as e:
            print(f"\n❌ Erreur dans le pipeline: {str(e)}")
            raise
    
    def generate_final_summary(self):
        """Générer un résumé final du pipeline"""
        print("\n" + "="*80)
        print("RÉSUMÉ FINAL DU PIPELINE CRISP-DM")
        print("="*80)
        
        print(f"\n📊 Données traitées:")
        if self.preprocessor:
            print(f"   - Données originales: {len(self.preprocessor.df)} lignes")
            print(f"   - Données nettoyées: {len(self.preprocessor.df_cleaned)} lignes")
            print(f"   - Features créées: {len(self.preprocessor.df_cleaned.columns)} colonnes")
        
        print(f"\n🤖 Modèles entraînés:")
        if self.detector:
            for model_name in self.detector.models.keys():
                anomalies = self.detector.anomaly_results[model_name]['predictions'].sum()
                total = len(self.detector.anomaly_results[model_name]['predictions'])
                pct = (anomalies / total) * 100
                print(f"   - {model_name}: {anomalies} anomalies ({pct:.2f}%)")
        
        print(f"\n📁 Fichiers générés:")
        print(f"   - Données nettoyées: {self.cleaned_data_path}")
        print(f"   - Résultats anomalies: {self.anomaly_results_path}")
        print(f"   - Graphiques: ../plots/")
        
        print(f"\n💡 Prochaines étapes recommandées:")
        print(f"   1. Analyser les anomalies détectées")
        print(f"   2. Valider les résultats avec des experts métier")
        print(f"   3. Mettre en place un système de monitoring")
        print(f"   4. Développer des alertes automatiques")

def main():
    """Fonction principale pour exécuter le pipeline complet"""
    
    # Configuration
    raw_data_path = Path("../datasets/reduced_hashed_input_data.csv")
    
    # Vérifier que le fichier existe
    if not raw_data_path.exists():
        print(f"❌ Fichier non trouvé: {raw_data_path}")
        print("Veuillez vous assurer que le fichier de données existe.")
        return
    
    # Initialiser le pipeline
    pipeline = CRISPDMPipeline(raw_data_path)
    
    # Exécuter le pipeline complet
    results = pipeline.run_complete_pipeline(
        exploration_sample=50000,    # Échantillon pour l'exploration
        preparation_sample=100000    # Échantillon pour la préparation
    )
    
    return results

if __name__ == "__main__":
    results = main()
