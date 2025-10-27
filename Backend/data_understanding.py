"""
Compréhension et Exploration des Données - CRISP-DM Phase 1
Dataset: reduced_hashed_input_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class BankingDataExplorer:
    """Classe pour l'exploration et la compréhension des données bancaires"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        
    def load_data(self, sample_size=None):
        """Charger les données avec option d'échantillonnage"""
        print("="*60)
        print("CHARGEMENT DES DONNÉES")
        print("="*60)
        
        if sample_size:
            # Charger un échantillon pour l'exploration
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
            
        print(f"✅ Données chargées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
        return self.df
    
    def basic_info(self):
        """Informations de base sur le dataset"""
        print("\n" + "="*60)
        print("INFORMATIONS DE BASE")
        print("="*60)
        
        print(f"\n📊 Dimensions: {self.df.shape}")
        print(f"💾 Taille mémoire: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n📋 Colonnes disponibles:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        print(f"\n🔍 Types de données:")
        print(self.df.dtypes)
        
        print(f"\n❓ Valeurs manquantes:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Colonne': missing.index,
            'Valeurs_manquantes': missing.values,
            'Pourcentage': missing_pct.values
        }).sort_values('Valeurs_manquantes', ascending=False)
        
        for _, row in missing_df.iterrows():
            if row['Valeurs_manquantes'] > 0:
                print(f"   {row['Colonne']}: {row['Valeurs_manquantes']} ({row['Pourcentage']:.1f}%)")
    
    def explore_categorical_variables(self):
        """Exploration des variables catégorielles"""
        print("\n" + "="*60)
        print("EXPLORATION DES VARIABLES CATÉGORIELLES")
        print("="*60)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            print(f"\n📊 {col}:")
            value_counts = self.df[col].value_counts()
            print(f"   Valeurs uniques: {self.df[col].nunique()}")
            print(f"   Top 5 valeurs:")
            for val, count in value_counts.head().items():
                pct = (count / len(self.df)) * 100
                print(f"     {val}: {count} ({pct:.1f}%)")
    
    def explore_numerical_variables(self):
        """Exploration des variables numériques"""
        print("\n" + "="*60)
        print("EXPLORATION DES VARIABLES NUMÉRIQUES")
        print("="*60)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            print(f"\n📊 Statistiques descriptives:")
            print(self.df[numerical_cols].describe())
        else:
            print("   Aucune variable numérique détectée")
    
    def explore_dates(self):
        """Exploration des variables de date"""
        print("\n" + "="*60)
        print("EXPLORATION DES VARIABLES DE DATE")
        print("="*60)
        
        # Identifier les colonnes qui pourraient être des dates
        potential_date_cols = []
        for col in self.df.columns:
            if any(keyword in col.upper() for keyword in ['DATE', 'DOU', 'DCO']):
                potential_date_cols.append(col)
        
        if potential_date_cols:
            for col in potential_date_cols:
                print(f"\n📅 {col}:")
                print(f"   Type actuel: {self.df[col].dtype}")
                print(f"   Exemples de valeurs:")
                print(f"     {self.df[col].head().tolist()}")
                
                # Essayer de convertir en datetime
                try:
                    date_converted = pd.to_datetime(self.df[col], errors='coerce')
                    valid_dates = date_converted.dropna()
                    if len(valid_dates) > 0:
                        print(f"   Conversion possible: {len(valid_dates)}/{len(self.df)} dates valides")
                        print(f"   Période: {valid_dates.min()} à {valid_dates.max()}")
                except:
                    print(f"   ❌ Conversion en date impossible")
        else:
            print("   Aucune colonne de date identifiée")
    
    def explore_clients(self):
        """Exploration des clients"""
        print("\n" + "="*60)
        print("EXPLORATION DES CLIENTS")
        print("="*60)
        
        if 'CLI' in self.df.columns:
            print(f"👥 Nombre de clients uniques: {self.df['CLI'].nunique()}")
            print(f"📊 Nombre moyen d'opérations par client: {len(self.df) / self.df['CLI'].nunique():.1f}")
            
            # Distribution des opérations par client
            operations_per_client = self.df['CLI'].value_counts()
            print(f"\n📈 Distribution des opérations par client:")
            print(f"   Min: {operations_per_client.min()}")
            print(f"   Max: {operations_per_client.max()}")
            print(f"   Médiane: {operations_per_client.median():.1f}")
            print(f"   Moyenne: {operations_per_client.mean():.1f}")
            
            # Top 5 clients avec le plus d'opérations
            print(f"\n🏆 Top 5 clients (nombre d'opérations):")
            for i, (client, count) in enumerate(operations_per_client.head().items(), 1):
                print(f"   {i}. Client {client}: {count} opérations")
        else:
            print("   ❌ Colonne 'CLI' non trouvée")
    
    def explore_transactions(self):
        """Exploration des transactions"""
        print("\n" + "="*60)
        print("EXPLORATION DES TRANSACTIONS")
        print("="*60)
        
        # Exploration des montants
        if 'MON' in self.df.columns:
            print(f"💰 Exploration des montants (MON):")
            print(f"   Type: {self.df['MON'].dtype}")
            print(f"   Valeurs uniques: {self.df['MON'].nunique()}")
            print(f"   Valeurs nulles: {self.df['MON'].isnull().sum()}")
            
            # Statistiques des montants
            mon_numeric = pd.to_numeric(self.df['MON'], errors='coerce')
            valid_amounts = mon_numeric.dropna()
            
            if len(valid_amounts) > 0:
                print(f"   Montants valides: {len(valid_amounts)}/{len(self.df)}")
                print(f"   Min: {valid_amounts.min():.2f}")
                print(f"   Max: {valid_amounts.max():.2f}")
                print(f"   Moyenne: {valid_amounts.mean():.2f}")
                print(f"   Médiane: {valid_amounts.median():.2f}")
        
        # Exploration des sens d'opération
        if 'SEN' in self.df.columns:
            print(f"\n🔄 Sens des opérations (SEN):")
            sen_counts = self.df['SEN'].value_counts()
            for sen, count in sen_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"   {sen}: {count} ({pct:.1f}%)")
        
        # Exploration des produits
        if 'CPRO' in self.df.columns:
            print(f"\n🏦 Produits (CPRO):")
            print(f"   Nombre de produits différents: {self.df['CPRO'].nunique()}")
            print(f"   Top 5 produits:")
            cpro_counts = self.df['CPRO'].value_counts()
            for i, (product, count) in enumerate(cpro_counts.head().items(), 1):
                pct = (count / len(self.df)) * 100
                print(f"     {i}. {product}: {count} ({pct:.1f}%)")
    
    def generate_visualizations(self, save_plots=True):
        """Générer des visualisations pour l'exploration"""
        print("\n" + "="*60)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("="*60)
        
        # Créer le dossier pour les graphiques
        plots_dir = Path("../plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Distribution des montants
        if 'MON' in self.df.columns:
            mon_numeric = pd.to_numeric(self.df['MON'], errors='coerce')
            valid_amounts = mon_numeric.dropna()
            
            if len(valid_amounts) > 0:
                plt.figure(figsize=(12, 8))
                
                # Graphique 1: Distribution des montants (log scale)
                plt.subplot(2, 2, 1)
                plt.hist(np.log10(valid_amounts + 1), bins=50, alpha=0.7, color='skyblue')
                plt.title('Distribution des Montants (Log Scale)')
                plt.xlabel('Log10(Montant + 1)')
                plt.ylabel('Fréquence')
                
                # Graphique 2: Box plot des montants
                plt.subplot(2, 2, 2)
                plt.boxplot(valid_amounts, vert=True)
                plt.title('Box Plot des Montants')
                plt.ylabel('Montant')
                
                # Graphique 3: Distribution des sens d'opération
                if 'SEN' in self.df.columns:
                    plt.subplot(2, 2, 3)
                    sen_counts = self.df['SEN'].value_counts()
                    plt.pie(sen_counts.values, labels=sen_counts.index, autopct='%1.1f%%')
                    plt.title('Répartition des Sens d\'Opération')
                
                # Graphique 4: Top 10 produits
                if 'CPRO' in self.df.columns:
                    plt.subplot(2, 2, 4)
                    cpro_counts = self.df['CPRO'].value_counts().head(10)
                    plt.bar(range(len(cpro_counts)), cpro_counts.values)
                    plt.title('Top 10 Produits')
                    plt.xlabel('Produits')
                    plt.ylabel('Nombre d\'opérations')
                    plt.xticks(range(len(cpro_counts)), cpro_counts.index, rotation=45)
                
                plt.tight_layout()
                if save_plots:
                    plt.savefig(plots_dir / 'data_exploration_overview.png', dpi=300, bbox_inches='tight')
                    print(f"✅ Graphique sauvegardé: {plots_dir / 'data_exploration_overview.png'}")
                plt.show()
        
        # 2. Timeline des opérations (si dates disponibles)
        date_cols = [col for col in self.df.columns if any(keyword in col.upper() for keyword in ['DATE', 'DOU', 'DCO'])]
        if date_cols:
            for col in date_cols:
                try:
                    dates = pd.to_datetime(self.df[col], errors='coerce').dropna()
                    if len(dates) > 0:
                        plt.figure(figsize=(12, 6))
                        dates.value_counts().sort_index().plot(kind='line')
                        plt.title(f'Timeline des Opérations - {col}')
                        plt.xlabel('Date')
                        plt.ylabel('Nombre d\'opérations')
                        plt.xticks(rotation=45)
                        
                        if save_plots:
                            plt.savefig(plots_dir / f'timeline_{col}.png', dpi=300, bbox_inches='tight')
                            print(f"✅ Timeline sauvegardée: {plots_dir / f'timeline_{col}.png'}")
                        plt.show()
                        break
                except:
                    continue
    
    def generate_summary_report(self):
        """Générer un rapport de synthèse de l'exploration"""
        print("\n" + "="*60)
        print("RAPPORT DE SYNTHÈSE - EXPLORATION DES DONNÉES")
        print("="*60)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   - Dimensions: {self.df.shape}")
        print(f"   - Mémoire: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"   - Colonnes: {len(self.df.columns)}")
        
        # Qualité des données
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        print(f"\n🔍 Qualité des données:")
        print(f"   - Valeurs manquantes: {missing_pct:.2f}%")
        print(f"   - Lignes complètes: {self.df.dropna().shape[0]} ({self.df.dropna().shape[0]/len(self.df)*100:.1f}%)")
        
        # Types de données
        print(f"\n📋 Types de données:")
        print(f"   - Numériques: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"   - Catégorielles: {len(self.df.select_dtypes(include=['object']).columns)}")
        print(f"   - Dates: {len([col for col in self.df.columns if any(keyword in col.upper() for keyword in ['DATE', 'DOU', 'DCO'])])}")
        
        # Recommandations
        print(f"\n💡 Recommandations pour la préparation:")
        print(f"   1. Nettoyer les valeurs manquantes")
        print(f"   2. Convertir les types de données appropriés")
        print(f"   3. Créer des features temporelles")
        print(f"   4. Standardiser les montants")
        print(f"   5. Encoder les variables catégorielles")

def main():
    """Fonction principale pour l'exploration des données"""
    
    # Initialisation
    dataset_path = Path("../datasets/reduced_hashed_input_data.csv")
    explorer = BankingDataExplorer(dataset_path)
    
    # Charger un échantillon pour l'exploration (50k lignes)
    explorer.load_data(sample_size=50000)
    
    # Exploration complète
    explorer.basic_info()
    explorer.explore_categorical_variables()
    explorer.explore_numerical_variables()
    explorer.explore_dates()
    explorer.explore_clients()
    explorer.explore_transactions()
    
    # Générer les visualisations
    explorer.generate_visualizations()
    
    # Rapport final
    explorer.generate_summary_report()
    
    return explorer.df

if __name__ == "__main__":
    df = main()
