import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model, model_name
    except Exception as e:
        print(f"⚠️ Échec du chargement de {model_name}: {e}")
        fallback = "tiiuae/falcon-7b-instruct"
        print(f"➡️ Basculer vers le fallback public {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForCausalLM.from_pretrained(fallback)
        return tokenizer, model, fallback

# Charger modèle (essayer Mistral gated, sinon fallback)
model_name_primary = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer, model, active_model = load_model(model_name_primary)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_llm_explanation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return f"[model: {active_model}] " + generated_text
