"""
Script de test pour vérifier le pipeline CRISP-DM
"""

import sys
from pathlib import Path

# Ajouter le répertoire Backend au path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Tester que tous les modules s'importent correctement"""
    print("🧪 Test des imports...")
    
    try:
        from data_understanding import BankingDataExplorer
        print("✅ data_understanding importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import data_understanding: {e}")
        return False
    
    try:
        from data_preparation import BankingDataPreprocessor
        print("✅ data_preparation importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import data_preparation: {e}")
        return False
    
    try:
        from anomaly_detection_ml import BankingAnomalyDetector
        print("✅ anomaly_detection_ml importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import anomaly_detection_ml: {e}")
        return False
    
    try:
        from crisp_dm_pipeline import CRISPDMPipeline
        print("✅ crisp_dm_pipeline importé avec succès")
    except Exception as e:
        print(f"❌ Erreur import crisp_dm_pipeline: {e}")
        return False
    
    return True

def test_data_path():
    """Tester que le fichier de données existe"""
    print("\n🧪 Test du fichier de données...")
    
    data_path = Path("../datasets/reduced_hashed_input_data.csv")
    
    if data_path.exists():
        print(f"✅ Fichier de données trouvé: {data_path}")
        return True
    else:
        print(f"❌ Fichier de données non trouvé: {data_path}")
        print("Veuillez vous assurer que le fichier existe.")
        return False

def test_environment():
    """Tester l'environnement Python"""
    print("\n🧪 Test de l'environnement...")
    
    try:
        import pandas as pd
        print(f"✅ pandas version: {pd.__version__}")
    except ImportError:
        print("❌ pandas non installé")
        return False
    
    try:
        import numpy as np
        print(f"✅ numpy version: {np.__version__}")
    except ImportError:
        print("❌ numpy non installé")
        return False
    
    try:
        import matplotlib
        print(f"✅ matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("❌ matplotlib non installé")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ seaborn version: {sns.__version__}")
    except ImportError:
        print("❌ seaborn non installé")
        return False
    
    try:
        import sklearn
        print(f"✅ scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn non installé")
        return False
    
    return True

def main():
    """Fonction principale de test"""
    print("🚀" + "="*60)
    print("TEST DU PIPELINE CRISP-DM")
    print("="*60)
    
    # Tests
    tests_passed = 0
    total_tests = 3
    
    if test_environment():
        tests_passed += 1
    
    if test_imports():
        tests_passed += 1
    
    if test_data_path():
        tests_passed += 1
    
    # Résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    
    print(f"Tests réussis: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests sont passés ! Le pipeline est prêt à être utilisé.")
        print("\n💡 Pour exécuter le pipeline complet:")
        print("   python crisp_dm_pipeline.py")
    else:
        print("❌ Certains tests ont échoué. Veuillez corriger les problèmes avant de continuer.")
        
        if tests_passed < 2:
            print("\n🔧 Solutions possibles:")
            print("   1. Installer les dépendances: pip install pandas numpy matplotlib seaborn scikit-learn")
            print("   2. Vérifier que vous êtes dans le bon répertoire")
            print("   3. Vérifier que le fichier de données existe")

if __name__ == "__main__":
    main()
