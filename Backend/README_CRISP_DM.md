# Pipeline CRISP-DM pour la Détection d'Anomalies Bancaires

## 📋 Structure du Projet

Ce projet suit la méthodologie CRISP-DM (Cross-Industry Standard Process for Data Mining) pour la détection d'anomalies dans les données bancaires.

### 🗂️ Fichiers du Pipeline

```
Backend/
├── data_understanding.py      # Phase 1: Exploration des données (EDA)
├── data_preparation.py        # Phase 2: Préparation et nettoyage
├── anomaly_detection_ml.py    # Phase 3: Détection d'anomalies (ML)
├── crisp_dm_pipeline.py       # Orchestrateur principal
├── test_pipeline.py           # Script de test du pipeline
├── run_pipeline.bat           # Script Windows (CMD)
├── run_pipeline.ps1           # Script Windows (PowerShell)
└── README_CRISP_DM.md         # Ce fichier
```

## 🚀 Utilisation

### Option 1: Pipeline Complet (Recommandé)

#### Windows (PowerShell)
```powershell
cd Backend
.\run_pipeline.ps1
```

#### Windows (Command Prompt)
```cmd
cd Backend
run_pipeline.bat
```

#### Linux/Mac
```bash
cd Backend
python crisp_dm_pipeline.py
```

### Option 2: Test et Pipeline Manuel

#### 1. Test de l'environnement
```bash
cd Backend
python test_pipeline.py
```

#### 2. Pipeline complet
```bash
python crisp_dm_pipeline.py
```

### Option 3: Étapes Individuelles

#### 1. Exploration des Données
```bash
python data_understanding.py
```

#### 2. Préparation des Données
```bash
python data_preparation.py
```

#### 3. Détection d'Anomalies
```bash
python anomaly_detection_ml.py
```

## 📊 Phases CRISP-DM

### Phase 1: Data Understanding (EDA)
**Fichier:** `data_understanding.py`

**Objectifs:**
- Comprendre la structure des données
- Identifier les patterns et distributions
- Détecter les problèmes de qualité
- Générer des visualisations exploratoires

**Sorties:**
- Rapport d'exploration
- Graphiques dans `../plots/`
- Recommandations pour la préparation

### Phase 2: Data Preparation
**Fichier:** `data_preparation.py`

**Objectifs:**
- Nettoyer les données
- Gérer les valeurs manquantes
- Créer des features dérivées
- Standardiser les formats

**Sorties:**
- `../datasets/cleaned_banking_data.csv`
- Rapport de nettoyage

### Phase 3: Modeling (Détection d'Anomalies)
**Fichier:** `anomaly_detection_ml.py`

**Objectifs:**
- Entraîner des modèles de ML non supervisé
- Détecter les anomalies
- Comparer les performances
- Générer des visualisations

**Modèles utilisés:**
- **Isolation Forest**: Détection d'outliers
- **One-Class SVM**: Classification binaire
- **DBSCAN**: Clustering avec détection d'anomalies

**Sorties:**
- `../datasets/anomaly_detection_results.csv`
- Graphiques de visualisation
- Rapport détaillé

## 🔧 Configuration

### Paramètres Modifiables

#### Dans `crisp_dm_pipeline.py`:
```python
# Taille des échantillons
exploration_sample=50000    # Pour l'EDA
preparation_sample=100000   # Pour la préparation
```

#### Dans `anomaly_detection_ml.py`:
```python
# Paramètres des modèles
contamination=0.1          # Isolation Forest
nu=0.1                     # One-Class SVM
eps=0.5, min_samples=5     # DBSCAN
```

## 📈 Résultats Attendus

### Données Nettoyées
- **Dimensions**: 100,000 lignes × 36 colonnes
- **Features créées**: 23 nouvelles variables
- **Qualité**: Données standardisées et prêtes pour le ML

### Anomalies Détectées
- **Isolation Forest**: ~10% des transactions
- **One-Class SVM**: ~10% des transactions
- **DBSCAN**: Variable selon les clusters
- **Consensus**: Anomalies validées par plusieurs modèles

## 🎯 Prochaines Étapes

1. **Validation Métier**: Analyser les anomalies détectées
2. **Optimisation**: Ajuster les paramètres des modèles
3. **Monitoring**: Mettre en place un système de surveillance
4. **Alertes**: Développer des notifications automatiques

## 📚 Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 🔍 Exemples de Features Créées

### Temporelles
- `AGE_COMPTE_JOURS`: Âge du compte en jours
- `MOIS_OPERATION`: Mois de l'opération
- `HEURE_OPERATION`: Heure de l'opération
- `EST_WEEKEND`: Flag weekend

### Comportementales
- `MONTANT_MOYEN`: Montant moyen par client
- `FREQUENCE_OPERATIONS`: Fréquence d'opérations
- `NB_PRODUITS_DIFFERENTS`: Nombre de produits

### Dérivées
- `MONTANT_RELATIF`: Montant relatif à la moyenne
- `ECART_STANDARD`: Écart en nombre d'écarts-types
- `CATEGORIE_MONTANT`: Catégorie du montant

## ⚠️ Notes Importantes

1. **Séparation des Phases**: Chaque phase est indépendante
2. **Pas de Détection dans Data Prep**: La détection d'anomalies est uniquement en Phase 3
3. **ML Non Supervisé**: Aucune donnée étiquetée requise
4. **Évolutif**: Facile d'ajouter de nouveaux modèles

## 🐛 Dépannage

### Erreurs Communes
- **Fichier non trouvé**: Vérifier le chemin vers `reduced_hashed_input_data.csv`
- **Mémoire insuffisante**: Réduire la taille des échantillons
- **Graphiques non affichés**: Vérifier l'installation de matplotlib

### Logs
Les scripts génèrent des logs détaillés pour chaque étape. Consultez la sortie console pour diagnostiquer les problèmes.
