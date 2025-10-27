"""
Préparation et nettoyage des données - CRISP-DM Phase 2
Dataset: reduced_hashed_input_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BankingDataPreprocessor:
    """Classe pour le nettoyage et la préparation des données bancaires"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.df_cleaned = None
        
    def load_data(self, sample_size=None):
        """Charger les données avec option d'échantillonnage"""
        print("Chargement des données...")
        
        if sample_size:
            # Charger un échantillon pour le développement
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(self.dataset_path, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) * chunk_size >= sample_size:
                    break
            self.df = pd.concat(chunks, ignore_index=True)
            if len(self.df) > sample_size:
                self.df = self.df.sample(n=sample_size, random_state=42)
        else:
            # Charger tout le dataset
            self.df = pd.read_csv(self.dataset_path)
            
        print(f"Données chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        return self.df
    
    def clean_data(self):
        """Nettoyer les données selon les bonnes pratiques bancaires"""
        print("\n" + "="*50)
        print("NETTOYAGE DES DONNÉES")
        print("="*50)
        
        df = self.df.copy()
        
        # 1. Gestion des valeurs manquantes
        print("\n1. Gestion des valeurs manquantes:")
        
        # DNA (Date de naissance) - 98% manquantes, on peut la supprimer
        if 'DNA' in df.columns:
            df = df.drop('DNA', axis=1)
            print("   - Colonne DNA supprimée (98% de valeurs manquantes)")
        
        # Nettoyage des espaces dans SEXT
        if 'SEXT' in df.columns:
            df['SEXT'] = df['SEXT'].str.strip()
            df['SEXT'] = df['SEXT'].replace('', np.nan)
            print("   - Espaces supprimés dans SEXT")
        
        # 2. Conversion des types de données
        print("\n2. Conversion des types de données:")
        
        # Conversion des dates
        date_columns = ['DOU', 'DCO']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f"   - {col} convertie en datetime")
        
        # Conversion des montants
        if 'MON' in df.columns:
            df['MON'] = pd.to_numeric(df['MON'], errors='coerce')
            print("   - MON converti en numérique")
        
        # 3. Nettoyage des libellés
        print("\n3. Nettoyage des libellés:")
        
        # Nettoyage des libellés d'opérations
        if 'BH_LIB' in df.columns:
            df['BH_LIB'] = df['BH_LIB'].str.strip()
            df['BH_LIB'] = df['BH_LIB'].str.replace(r'\s+', ' ', regex=True)
            print("   - BH_LIB nettoyé (espaces multiples supprimés)")
        
        # 4. Création de nouvelles variables dérivées
        print("\n4. Création de variables dérivées:")
        
        # Âge du compte en jours
        if 'DOU' in df.columns and 'DCO' in df.columns:
            df['AGE_COMPTE_JOURS'] = (df['DCO'] - df['DOU']).dt.days
            df['AGE_COMPTE_ANNEES'] = df['AGE_COMPTE_JOURS'] / 365.25
            print("   - AGE_COMPTE_JOURS et AGE_COMPTE_ANNEES créés")
        
        # Mois et jour de la semaine de l'opération
        if 'DCO' in df.columns:
            df['MOIS_OPERATION'] = df['DCO'].dt.month
            df['JOUR_SEMAINE'] = df['DCO'].dt.dayofweek
            df['HEURE_OPERATION'] = df['DCO'].dt.hour
            print("   - Variables temporelles créées (mois, jour, heure)")
        
        # Catégorisation des montants
        if 'MON' in df.columns:
            df['MONTANT_ABSOLU'] = df['MON'].abs()
            df['CATEGORIE_MONTANT'] = pd.cut(df['MONTANT_ABSOLU'], 
                                           bins=[0, 100, 500, 1000, 5000, float('inf')],
                                           labels=['Très petit', 'Petit', 'Moyen', 'Grand', 'Très grand'])
            print("   - Catégorisation des montants créée")
        
        # 5. Suppression des lignes avec des données critiques manquantes
        print("\n5. Suppression des lignes avec données critiques manquantes:")
        initial_rows = len(df)
        
        # Supprimer les lignes sans montant ou sans date
        df = df.dropna(subset=['MON', 'DCO'])
        
        # Supprimer les montants nuls
        df = df[df['MON'] != 0]
        
        final_rows = len(df)
        print(f"   - Lignes supprimées: {initial_rows - final_rows}")
        print(f"   - Lignes restantes: {final_rows}")
        
        self.df_cleaned = df
        return df
    
    def create_features(self):
        """Créer des features pour l'analyse des données"""
        print("\n" + "="*50)
        print("CRÉATION DE FEATURES POUR L'ANALYSE")
        print("="*50)
        
        df = self.df_cleaned.copy()
        
        # 1. Features temporelles
        print("\n1. Features temporelles:")
        
        if 'DCO' in df.columns:
            # Période de la journée
            df['PERIODE_JOUR'] = pd.cut(df['DCO'].dt.hour, 
                                       bins=[0, 6, 12, 18, 24], 
                                       labels=['Nuit', 'Matin', 'Après-midi', 'Soir'],
                                       include_lowest=True)
            
            # Weekend vs semaine
            df['EST_WEEKEND'] = df['DCO'].dt.dayofweek.isin([5, 6])
            
            # Jour du mois (pour détecter les patterns de fin de mois)
            df['JOUR_MOIS'] = df['DCO'].dt.day
            
            print("   - PERIODE_JOUR, EST_WEEKEND, JOUR_MOIS créés")
        
        # 2. Features comportementales par client
        print("\n2. Features comportementales par client:")
        
        if 'CLI' in df.columns:
            # Statistiques par client
            client_stats = df.groupby('CLI').agg({
                'MON': ['count', 'mean', 'std', 'min', 'max'],
                'DCO': ['min', 'max']
            }).round(2)
            
            # Aplatir les colonnes
            client_stats.columns = ['_'.join(col).strip() for col in client_stats.columns]
            client_stats = client_stats.reset_index()
            
            # Renommer les colonnes
            client_stats.columns = ['CLI', 'NB_OPERATIONS', 'MONTANT_MOYEN', 'MONTANT_STD', 
                                  'MONTANT_MIN', 'MONTANT_MAX', 'PREMIERE_OPERATION', 'DERNIERE_OPERATION']
            
            # Calculer la fréquence d'opérations
            client_stats['FREQUENCE_OPERATIONS'] = client_stats['NB_OPERATIONS'] / (
                (client_stats['DERNIERE_OPERATION'] - client_stats['PREMIERE_OPERATION']).dt.days + 1
            )
            
            # Joindre avec le dataset principal
            df = df.merge(client_stats, on='CLI', how='left')
            
            print("   - Statistiques comportementales par client créées")
        
        # 3. Features de montant
        print("\n3. Features de montant:")
        
        if 'MON' in df.columns and 'CLI' in df.columns:
            # Montant relatif par rapport à la moyenne du client
            df['MONTANT_RELATIF'] = df['MON'] / df['MONTANT_MOYEN']
            
            # Écart par rapport à la moyenne (en nombre d'écarts-types)
            df['ECART_STANDARD'] = (df['MON'] - df['MONTANT_MOYEN']) / df['MONTANT_STD']
            df['ECART_STANDARD'] = df['ECART_STANDARD'].fillna(0)
            
            print("   - MONTANT_RELATIF et ECART_STANDARD créés")
        
        # 4. Features de produits
        print("\n4. Features de produits:")
        
        if 'CPRO' in df.columns and 'CLI' in df.columns:
            # Nombre de produits différents par client
            produits_par_client = df.groupby('CLI')['CPRO'].nunique().reset_index()
            produits_par_client.columns = ['CLI', 'NB_PRODUITS_DIFFERENTS']
            df = df.merge(produits_par_client, on='CLI', how='left')
            
            print("   - NB_PRODUITS_DIFFERENTS créé")
        
        # 5. Features de comportement (pour analyse future)
        print("\n5. Features de comportement:")
        
        # Flag pour les opérations en dehors des heures normales
        if 'HEURE_OPERATION' in df.columns:
            df['HEURE_ANORMALE'] = ((df['HEURE_OPERATION'] < 6) | (df['HEURE_OPERATION'] > 22)).astype(int)
        
        # Flag pour les opérations le weekend
        if 'EST_WEEKEND' in df.columns:
            df['OPERATION_WEEKEND'] = df['EST_WEEKEND'].astype(int)
        
        print("   - Features de comportement créées")
        
        self.df_cleaned = df
        return df
    
    def generate_summary_report(self):
        """Générer un rapport de synthèse du nettoyage"""
        print("\n" + "="*60)
        print("RAPPORT DE SYNTHÈSE - DONNÉES NETTOYÉES")
        print("="*60)
        
        df = self.df_cleaned
        
        print(f"\nDimensions finales: {df.shape}")
        print(f"Taille mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Statistiques par client
        if 'CLI' in df.columns:
            print(f"\nNombre de clients uniques: {df['CLI'].nunique()}")
            print(f"Nombre moyen d'opérations par client: {df.groupby('CLI').size().mean():.1f}")
        
        # Statistiques des montants
        if 'MON' in df.columns:
            print(f"\nStatistiques des montants:")
            print(f"  - Montant moyen: {df['MON'].mean():.2f}")
            print(f"  - Montant médian: {df['MON'].median():.2f}")
            print(f"  - Montant min: {df['MON'].min():.2f}")
            print(f"  - Montant max: {df['MON'].max():.2f}")
        
        # Répartition des opérations
        if 'SEN' in df.columns:
            print(f"\nRépartition des opérations:")
            print(df['SEN'].value_counts())
        
        # Période des données
        if 'DCO' in df.columns:
            print(f"\nPériode des données:")
            print(f"  - Du: {df['DCO'].min()}")
            print(f"  - Au: {df['DCO'].max()}")
        
        # Features de comportement
        if 'HEURE_ANORMALE' in df.columns:
            heures_anormales = df['HEURE_ANORMALE'].sum()
            print(f"\nOpérations en heures anormales: {heures_anormales} ({heures_anormales/len(df)*100:.2f}%)")
        
        if 'OPERATION_WEEKEND' in df.columns:
            operations_weekend = df['OPERATION_WEEKEND'].sum()
            print(f"Opérations le weekend: {operations_weekend} ({operations_weekend/len(df)*100:.2f}%)")
        
        return df
    
    def save_cleaned_data(self, output_path):
        """Sauvegarder les données nettoyées"""
        print(f"\nSauvegarde des données nettoyées vers: {output_path}")
        self.df_cleaned.to_csv(output_path, index=False)
        print("Sauvegarde terminée!")

def main():
    """Fonction principale pour le nettoyage des données"""
    
    # Initialisation
    dataset_path = Path("../datasets/reduced_hashed_input_data.csv")
    preprocessor = BankingDataPreprocessor(dataset_path)
    
    # Charger un échantillon pour le développement (100k lignes)
    preprocessor.load_data(sample_size=100000)
    
    # Nettoyer les données
    preprocessor.clean_data()
    
    # Créer les features
    preprocessor.create_features()
    
    # Générer le rapport
    preprocessor.generate_summary_report()
    
    # Sauvegarder
    output_path = Path("../datasets/cleaned_banking_data.csv")
    preprocessor.save_cleaned_data(output_path)
    
    return preprocessor.df_cleaned

if __name__ == "__main__":
    cleaned_df = main()
