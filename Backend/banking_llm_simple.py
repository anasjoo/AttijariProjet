import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class BankingLLM:
    def __init__(self):
        self.device = "cpu"
        self.model_name = "tiiuae/falcon-7b-instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def expliquer_transaction(self, client_id, profession, revenu, transaction):
        """Génère une explication améliorée pour une transaction suspecte"""
        
        # Extraire le montant de la transaction
        montant_match = re.search(r'(\d+(?:\.\d+)?)', transaction)
        montant = float(montant_match.group(1)) if montant_match else 0
        ratio = round(montant / revenu, 2) if revenu > 0 else 0
        
        prompt = f"""En tant qu'expert bancaire, analysez cette transaction en tenant compte du revenu mensuel du client.

Client: {profession} avec un revenu mensuel de {revenu} TND
Transaction: {transaction}
Analyse financière: Le montant de {montant} TND représente {ratio} fois le revenu mensuel du client.

Expliquez en 2-3 phrases simples pourquoi cette transaction est inhabituelle ou suspecte :"""
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                min_length=30,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explication = response.replace(prompt, "").strip()
        
        # Nettoyage
        explication = explication.split("\n")[0]
        if len(explication) > 200:
            explication = explication[:200] + "..."
            
        return explication

# Test rapide
if __name__ == "__main__":
    llm = BankingLLM()
    
    print("🧪 Test 1:")
    result1 = llm.expliquer_transaction(
        client_id="12345",
        profession="Ingénieur",
        revenu=3500,
        transaction="Retrait de 5000 TND à 3h du matin"
    )
    print(f"Explication: {result1}\n")
    
    print("🧪 Test 2:")
    result2 = llm.expliquer_transaction(
        client_id="67890",
        profession="Enseignant",
        revenu=1800,
        transaction="Virement de 10000 TND vers compte étranger"
    )
    print(f"Explication: {result2}")
