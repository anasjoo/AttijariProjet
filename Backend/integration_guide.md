# Guide d'Intégration - Banking LLM Simple

## 🚀 Utilisation Immédiate

### 1. Test Rapide
```bash
cd Backend
python banking_llm_simple.py
```

### 2. Intégration dans votre code existant

#### **Option A : Remplacement direct**
```python
# Dans votre fichier actuel (ex: anomaly_detector.py)
from banking_llm_simple import BankingLLM

# Créer une instance
llm = BankingLLM()

# Utiliser pour générer des explications
explication = llm.expliquer_transaction(
    client_id="12345",
    profession="Ingénieur", 
    revenu=3500,
    transaction="Retrait de 5000 TND à 3h du matin"
)
```

#### **Option B : Intégration avec vos données**
```python
# Exemple avec vos structures de données actuelles
def generer_explication_anomalie(client_data, transaction_data):
    llm = BankingLLM()
    
    return llm.expliquer_transaction(
        client_id=client_data['client_id'],
        profession=client_data['profession'],
        revenu=client_data['revenu_mensuel'],
        transaction=f"{transaction_data['type']} de {transaction_data['montant']} TND - {transaction_data['details']}"
    )

# Utilisation
explication = generer_explication_anomalie(client, transaction)
```

### 3. Intégration dans votre pipeline existant

#### **Dans `anomaly_detector.py` :**
```python
# Ajouter en haut du fichier
from banking_llm_simple import BankingLLM

# Dans votre fonction de détection
def detect_and_explain_anomalies(client_data, transactions):
    llm = BankingLLM()
    anomalies = []
    
    for transaction in transactions:
        if is_anomaly(transaction, client_data):
            explication = llm.expliquer_transaction(
                client_id=client_data['client_id'],
                profession=client_data['profession'],
                revenu=client_data['revenu_mensuel'],
                transaction=f"{transaction['type']} de {transaction['montant']} TND"
            )
            
            anomalies.append({
                'transaction': transaction,
                'explication': explication
            })
    
    return anomalies
```

#### **Dans `main.py` :**
```python
# Remplacer votre appel LLM actuel par :
from banking_llm_simple import BankingLLM

# Au lieu de :
# response = explainer.generate_response(prompt)

# Utiliser :
llm = BankingLLM()
explication = llm.expliquer_transaction(
    client_id=client['id'],
    profession=client['profession'],
    revenu=client['revenu'],
    transaction=transaction_description
)
```

### 4. Exemples d'utilisation

#### **Cas 1 : Analyse de transaction suspecte**
```python
llm = BankingLLM()
resultat = llm.expliquer_transaction(
    client_id="C001",
    profession="Comptable",
    revenu=2500,
    transaction="Retrait de 8000 TND à 2h du matin"
)
# Résultat : "Cette transaction est inhabituelle car le retrait de 8000 TND représente plus de 3 fois le revenu mensuel du client..."
```

#### **Cas 2 : Virement inhabituel**
```python
resultat = llm.expliquer_transaction(
    client_id="C002",
    profession="Enseignant",
    revenu=1800,
    transaction="Virement de 15000 TND vers un compte inconnu"
)
# Résultat : "Ce virement est suspect car il dépasse largement les capacités financières habituelles..."
```

### 5. Avantages de cette solution
- ✅ **Simple** : Une seule ligne d'appel
- ✅ **Propre** : Pas d'avertissements techniques
- ✅ **Rapide** : Optimisé pour la performance
- ✅ **Compatible** : Fonctionne avec vos données existantes
- ✅ **Français** : Réponses en français clair et professionnel

### 6. Migration depuis votre code actuel

#### **Étape 1 : Identifier où vous utilisez l'ancien LLM**
Regardez dans vos fichiers :
- `test_prompt.py`
- `anomaly_detector.py`
- `main.py`

#### **Étape 2 : Remplacer l'import**
```python
# Remplacer :
# from llm_explainer import LLMExplainer

# Par :
from banking_llm_simple import BankingLLM
```

#### **Étape 3 : Remplacer l'appel**
```python
# Remplacer :
# response = explainer.generate_response(prompt)

# Par :
llm = BankingLLM()
response = llm.expliquer_transaction(
    client_id=client_id,
    profession=profession,
    revenu=revenu,
    transaction=description_transaction
)
```

### 7. Test d'intégration
```python
# Test rapide avec vos données
from banking_llm_simple import BankingLLM

llm = BankingLLM()
print("Test d'intégration réussi !")
print(llm.expliquer_transaction("TEST", "Test", 1000, "Test transaction"))
