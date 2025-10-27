from llm_explainer import get_llm_explanation

with open("prompts/explication_anomaly.txt", "r", encoding="utf-8") as f:
    for line in f:
        prompt = line.strip()
        if prompt:
            print("🔹 Prompt:", prompt)
            print("🧠 Réponse:", get_llm_explanation(prompt))
            print("------")