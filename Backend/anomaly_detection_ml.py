"""
Détection d'Anomalies avec Machine Learning Non Supervisé - CRISP-DM Phase 4
Dataset: cleaned_banking_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Imports pour le ML
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix

class BankingAnomalyDetector:
    """Classe pour la détection d'anomalies dans les données bancaires"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.anomaly_results = {}
        
    def load_data(self):
        """Charger les données nettoyées"""
        print("="*60)
        print("CHARGEMENT DES DONNÉES NETTOYÉES")
        print("="*60)
        
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Données chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        
        return self.df
    
    def prepare_features_for_ml(self):
        """Préparer les features pour le machine learning"""
        print("\n" + "="*60)
        print("PRÉPARATION DES FEATURES POUR LE ML")
        print("="*60)
        
        df = self.df.copy()
        
        # 1. Sélectionner les features numériques
        print("\n1. Sélection des features numériques:")
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure les colonnes non pertinentes
        exclude_cols = ['CLI']  # ID client
        numerical_features = [col for col in numerical_features if col not in exclude_cols]
        
        print(f"   Features numériques sélectionnées: {len(numerical_features)}")
        for i, feature in enumerate(numerical_features, 1):
            print(f"     {i:2d}. {feature}")
        
        # 2. Encoder les variables catégorielles
        print("\n2. Encodage des variables catégorielles:")
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"   - {col} encodé en {col}_encoded")
        
        # 3. Créer le dataset final pour le ML
        ml_features = numerical_features + [f'{col}_encoded' for col in categorical_features if col in df.columns]
        
        # Supprimer les valeurs manquantes
        df_ml = df[ml_features].dropna()
        
        print(f"\n3. Dataset final pour le ML:")
        print(f"   - Features: {len(ml_features)}")
        print(f"   - Lignes: {len(df_ml)}")
        print(f"   - Valeurs manquantes: {df_ml.isnull().sum().sum()}")
        
        self.df_processed = df_ml
        return df_ml, ml_features
    
    def train_isolation_forest(self, contamination=0.1):
        """Entraîner le modèle Isolation Forest"""
        print("\n" + "="*60)
        print("ENTRAÎNEMENT ISOLATION FOREST")
        print("="*60)
        
        # Standardiser les données
        X_scaled = self.scaler.fit_transform(self.df_processed)
        
        # Entraîner le modèle
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X_scaled)
        
        # Prédictions
        predictions = model.fit_predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        # Convertir les prédictions (-1 = anomalie, 1 = normal)
        anomaly_labels = (predictions == -1).astype(int)
        
        self.models['isolation_forest'] = model
        self.anomaly_results['isolation_forest'] = {
            'predictions': anomaly_labels,
            'scores': anomaly_scores,
            'contamination': contamination
        }
        
        print(f"✅ Isolation Forest entraîné")
        print(f"   - Contamination: {contamination}")
        print(f"   - Anomalies détectées: {anomaly_labels.sum()} ({anomaly_labels.sum()/len(anomaly_labels)*100:.2f}%)")
        
        return model, anomaly_labels, anomaly_scores
    
    def train_one_class_svm(self, nu=0.1):
        """Entraîner le modèle One-Class SVM"""
        print("\n" + "="*60)
        print("ENTRAÎNEMENT ONE-CLASS SVM")
        print("="*60)
        
        # Standardiser les données
        X_scaled = self.scaler.fit_transform(self.df_processed)
        
        # Entraîner le modèle
        model = OneClassSVM(
            nu=nu,
            kernel='rbf',
            gamma='scale'
        )
        
        model.fit(X_scaled)
        
        # Prédictions
        predictions = model.predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        
        # Convertir les prédictions (-1 = anomalie, 1 = normal)
        anomaly_labels = (predictions == -1).astype(int)
        
        self.models['one_class_svm'] = model
        self.anomaly_results['one_class_svm'] = {
            'predictions': anomaly_labels,
            'scores': anomaly_scores,
            'nu': nu
        }
        
        print(f"✅ One-Class SVM entraîné")
        print(f"   - Nu: {nu}")
        print(f"   - Anomalies détectées: {anomaly_labels.sum()} ({anomaly_labels.sum()/len(anomaly_labels)*100:.2f}%)")
        
        return model, anomaly_labels, anomaly_scores
    
    def train_dbscan(self, eps=0.5, min_samples=5):
        """Entraîner le modèle DBSCAN pour la détection d'anomalies"""
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DBSCAN")
        print("="*60)
        
        # Standardiser les données
        X_scaled = self.scaler.fit_transform(self.df_processed)
        
        # Entraîner le modèle
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )
        
        cluster_labels = model.fit_predict(X_scaled)
        
        # Les points avec label -1 sont considérés comme des anomalies
        anomaly_labels = (cluster_labels == -1).astype(int)
        
        self.models['dbscan'] = model
        self.anomaly_results['dbscan'] = {
            'predictions': anomaly_labels,
            'cluster_labels': cluster_labels,
            'eps': eps,
            'min_samples': min_samples
        }
        
        print(f"✅ DBSCAN entraîné")
        print(f"   - Eps: {eps}")
        print(f"   - Min samples: {min_samples}")
        print(f"   - Anomalies détectées: {anomaly_labels.sum()} ({anomaly_labels.sum()/len(anomaly_labels)*100:.2f}%)")
        print(f"   - Clusters trouvés: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
        
        return model, anomaly_labels, cluster_labels
    
    def analyze_anomalies(self):
        """Analyser les anomalies détectées"""
        print("\n" + "="*60)
        print("ANALYSE DES ANOMALIES DÉTECTÉES")
        print("="*60)
        
        # Créer un DataFrame avec les résultats
        results_df = self.df_processed.copy()
        
        # Ajouter les prédictions de chaque modèle
        for model_name, results in self.anomaly_results.items():
            results_df[f'anomaly_{model_name}'] = results['predictions']
            if 'scores' in results:
                results_df[f'score_{model_name}'] = results['scores']
        
        # Analyse des anomalies par modèle
        print("\n📊 Résumé des anomalies par modèle:")
        for model_name, results in self.anomaly_results.items():
            anomalies = results['predictions'].sum()
            total = len(results['predictions'])
            pct = (anomalies / total) * 100
            print(f"   {model_name}: {anomalies} anomalies ({pct:.2f}%)")
        
        # Analyse des anomalies communes
        if len(self.anomaly_results) > 1:
            print("\n🔍 Analyse des anomalies communes:")
            
            # Créer une matrice de consensus
            consensus_cols = [f'anomaly_{name}' for name in self.anomaly_results.keys()]
            consensus_matrix = results_df[consensus_cols]
            
            # Compter les votes
            consensus_matrix['vote_count'] = consensus_matrix.sum(axis=1)
            consensus_matrix['consensus'] = (consensus_matrix['vote_count'] >= len(self.anomaly_results) // 2 + 1).astype(int)
            
            consensus_anomalies = consensus_matrix['consensus'].sum()
            print(f"   Anomalies par consensus: {consensus_anomalies} ({consensus_anomalies/len(results_df)*100:.2f}%)")
            
            # Distribution des votes
            vote_distribution = consensus_matrix['vote_count'].value_counts().sort_index()
            print(f"   Distribution des votes:")
            for votes, count in vote_distribution.items():
                print(f"     {votes} vote(s): {count} transactions")
        
        return results_df
    
    def visualize_anomalies(self, save_plots=True):
        """Visualiser les anomalies détectées"""
        print("\n" + "="*60)
        print("VISUALISATION DES ANOMALIES")
        print("="*60)
        
        # Créer le dossier pour les graphiques
        plots_dir = Path("../plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Réduction de dimension avec PCA
        X_scaled = self.scaler.fit_transform(self.df_processed)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 2. Graphiques pour chaque modèle
        n_models = len(self.anomaly_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_name, results) in enumerate(self.anomaly_results.items()):
            anomalies = results['predictions']
            
            # Graphique 1: Scatter plot avec PCA
            ax1 = axes[0, i] if n_models > 1 else axes[0]
            scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=anomalies, cmap='RdYlBu', alpha=0.6)
            ax1.set_title(f'Anomalies - {model_name}')
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, ax=ax1)
            
            # Graphique 2: Distribution des scores (si disponible)
            ax2 = axes[1, i] if n_models > 1 else axes[1]
            if 'scores' in results:
                scores = results['scores']
                ax2.hist(scores, bins=50, alpha=0.7, color='skyblue')
                ax2.axvline(np.percentile(scores, 10), color='red', linestyle='--', label='Seuil 10%')
                ax2.set_title(f'Distribution des scores - {model_name}')
                ax2.set_xlabel('Score')
                ax2.set_ylabel('Fréquence')
                ax2.legend()
            else:
                # Pour DBSCAN, montrer la distribution des clusters
                if 'cluster_labels' in results:
                    cluster_labels = results['cluster_labels']
                    unique_labels = np.unique(cluster_labels)
                    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                    
                    for label, color in zip(unique_labels, colors):
                        if label == -1:
                            # Anomalies en noir
                            ax2.scatter(X_pca[cluster_labels == label, 0], 
                                      X_pca[cluster_labels == label, 1], 
                                      c='black', marker='x', s=50, label='Anomalies')
                        else:
                            ax2.scatter(X_pca[cluster_labels == label, 0], 
                                      X_pca[cluster_labels == label, 1], 
                                      c=color, label=f'Cluster {label}')
                    
                    ax2.set_title(f'Clusters - {model_name}')
                    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                    ax2.legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / 'anomaly_detection_results.png', dpi=300, bbox_inches='tight')
            print(f"✅ Graphiques sauvegardés: {plots_dir / 'anomaly_detection_results.png'}")
        plt.show()
        
        # 3. Graphique de comparaison des modèles
        if len(self.anomaly_results) > 1:
            plt.figure(figsize=(12, 8))
            
            # Matrice de corrélation des prédictions
            consensus_cols = [f'anomaly_{name}' for name in self.anomaly_results.keys()]
            consensus_matrix = pd.DataFrame({
                name: results['predictions'] 
                for name, results in self.anomaly_results.items()
            })
            
            correlation_matrix = consensus_matrix.corr()
            
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Corrélation entre les prédictions des modèles')
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(plots_dir / 'model_correlation.png', dpi=300, bbox_inches='tight')
                print(f"✅ Matrice de corrélation sauvegardée: {plots_dir / 'model_correlation.png'}")
            plt.show()
    
    def generate_anomaly_report(self):
        """Générer un rapport détaillé des anomalies"""
        print("\n" + "="*60)
        print("RAPPORT DÉTAILLÉ DES ANOMALIES")
        print("="*60)
        
        # Créer un DataFrame avec les résultats
        results_df = self.df_processed.copy()
        
        # Ajouter les prédictions
        for model_name, results in self.anomaly_results.items():
            results_df[f'anomaly_{model_name}'] = results['predictions']
        
        # Statistiques générales
        print(f"\n📊 Statistiques générales:")
        print(f"   - Total des transactions: {len(results_df)}")
        print(f"   - Nombre de modèles: {len(self.anomaly_results)}")
        
        # Statistiques par modèle
        print(f"\n🔍 Détails par modèle:")
        for model_name, results in self.anomaly_results.items():
            anomalies = results['predictions'].sum()
            total = len(results['predictions'])
            pct = (anomalies / total) * 100
            
            print(f"\n   {model_name.upper()}:")
            print(f"     - Anomalies détectées: {anomalies}")
            print(f"     - Pourcentage: {pct:.2f}%")
            
            if 'scores' in results:
                scores = results['scores']
                print(f"     - Score moyen: {scores.mean():.3f}")
                print(f"     - Score min: {scores.min():.3f}")
                print(f"     - Score max: {scores.max():.3f}")
        
        # Anomalies par consensus
        if len(self.anomaly_results) > 1:
            print(f"\n🤝 Analyse par consensus:")
            consensus_cols = [f'anomaly_{name}' for name in self.anomaly_results.keys()]
            consensus_matrix = results_df[consensus_cols]
            
            # Compter les votes
            consensus_matrix['vote_count'] = consensus_matrix.sum(axis=1)
            
            for threshold in [1, len(self.anomaly_results)//2 + 1, len(self.anomaly_results)]:
                consensus_anomalies = (consensus_matrix['vote_count'] >= threshold).sum()
                pct = (consensus_anomalies / len(results_df)) * 100
                print(f"     - ≥{threshold} vote(s): {consensus_anomalies} anomalies ({pct:.2f}%)")
        
        return results_df
    
    def save_results(self, output_path):
        """Sauvegarder les résultats de détection d'anomalies"""
        print(f"\n💾 Sauvegarde des résultats vers: {output_path}")
        
        # Créer le DataFrame final
        results_df = self.df_processed.copy()
        
        # Ajouter les prédictions et scores
        for model_name, results in self.anomaly_results.items():
            results_df[f'anomaly_{model_name}'] = results['predictions']
            if 'scores' in results:
                results_df[f'score_{model_name}'] = results['scores']
        
        # Ajouter le consensus
        if len(self.anomaly_results) > 1:
            consensus_cols = [f'anomaly_{name}' for name in self.anomaly_results.keys()]
            consensus_matrix = results_df[consensus_cols]
            consensus_matrix['vote_count'] = consensus_matrix.sum(axis=1)
            consensus_matrix['consensus'] = (consensus_matrix['vote_count'] >= len(self.anomaly_results) // 2 + 1).astype(int)
            
            results_df['anomaly_consensus'] = consensus_matrix['consensus']
            results_df['anomaly_vote_count'] = consensus_matrix['vote_count']
        
        # Sauvegarder
        results_df.to_csv(output_path, index=False)
        print("✅ Sauvegarde terminée!")

def main():
    """Fonction principale pour la détection d'anomalies"""
    
    # Initialisation
    dataset_path = Path("../datasets/cleaned_banking_data.csv")
    detector = BankingAnomalyDetector(dataset_path)
    
    # Charger les données
    detector.load_data()
    
    # Préparer les features
    df_ml, features = detector.prepare_features_for_ml()
    
    # Entraîner les modèles
    detector.train_isolation_forest(contamination=0.1)
    detector.train_one_class_svm(nu=0.1)
    detector.train_dbscan(eps=0.5, min_samples=5)
    
    # Analyser les anomalies
    results_df = detector.analyze_anomalies()
    
    # Visualiser les résultats
    detector.visualize_anomalies()
    
    # Générer le rapport
    detector.generate_anomaly_report()
    
    # Sauvegarder les résultats
    output_path = Path("../datasets/anomaly_detection_results.csv")
    detector.save_results(output_path)
    
    return detector, results_df

if __name__ == "__main__":
    detector, results = main()
