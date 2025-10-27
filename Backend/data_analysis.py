"""
Analyse du dataset réel de la banque - reduced_hashed_input_data.csv
Suivant la méthodologie CRISP-DM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """Analyse complète du dataset selon CRISP-DM"""
    
    print("=" * 60)
    print("ANALYSE DU DATASET RÉEL DE LA BANQUE")
    print("=" * 60)
    
    # Chemin vers le dataset
    dataset_path = Path("../datasets/reduced_hashed_input_data.csv")
    
    # 1. COMPRÉHENSION DES DONNÉES (Data Understanding)
    print("\n1. COMPRÉHENSION DES DONNÉES")
    print("-" * 40)
    
    # Lecture par chunks pour gérer le gros fichier
    print("Lecture du dataset...")
    chunk_size = 10000
    chunks = []
    
    try:
        for chunk in pd.read_csv(dataset_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) >= 10:  # Limiter à 100k lignes pour l'analyse initiale
                break
    except Exception as e:
        print(f"Erreur lors de la lecture: {e}")
        return
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Informations générales
    print(f"\nDimensions: {df.shape}")
    print(f"Taille mémoire: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Structure des données
    print("\nColonnes du dataset:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    # Types de données
    print("\nTypes de données:")
    print(df.dtypes)
    
    # Statistiques descriptives
    print("\nStatistiques descriptives:")
    print(df.describe(include='all'))
    
    # Valeurs manquantes
    print("\nValeurs manquantes:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Colonne': missing_data.index,
        'Valeurs_manquantes': missing_data.values,
        'Pourcentage': missing_percent.values
    })
    missing_df = missing_df[missing_df['Valeurs_manquantes'] > 0].sort_values('Valeurs_manquantes', ascending=False)
    print(missing_df)
    
    # Échantillon des données
    print("\nÉchantillon des données (5 premières lignes):")
    print(df.head())
    
    # Analyse des colonnes catégorielles
    print("\nAnalyse des colonnes catégorielles:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} valeurs uniques")
        if unique_count <= 20:
            print(f"  Valeurs: {df[col].unique()}")
        else:
            print(f"  Premières valeurs: {df[col].unique()[:10]}")
    
    # Sauvegarde des informations pour la suite
    analysis_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_data': missing_df.to_dict('records'),
        'categorical_cols': list(categorical_cols),
        'sample_data': df.head().to_dict('records')
    }
    
    return df, analysis_info

if __name__ == "__main__":
    df, info = analyze_dataset()


