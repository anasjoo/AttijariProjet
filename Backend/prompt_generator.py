# def generate_prompt(client_row, transaction_row):
#     return f"""
# Client {client_row['client_id']} | Profession : {client_row['profession']} | Revenu mensuel : {client_row['revenu_mensuel']} TND
# Transaction suspecte :
# - Montant : {transaction_row['montant']} TND
# - Date : {transaction_row['date'].strftime('%Y-%m-%d')}
# - Type : {transaction_row['type']}
# - Lieu : {transaction_row['lieu']}
# - Canal : {transaction_row['canal']}

# Explique pourquoi cette transaction pourrait être inhabituelle.
# """

import os

def load_template(template_name):
    with open(os.path.join("prompts", template_name), "r", encoding="utf-8") as f:
        return f.read()

def generate_prompt_anomalie(client_row, tx_row):
    template = load_template("explication_anomaly.txt")
    return template.format(
client_id=client_row["client_id"],
profession=client_row["profession"],
revenu=client_row["revenu_mensuel"],
montant=tx_row["montant"],
date=tx_row["date"].strftime("%Y-%m-%d"),
type=tx_row["type"],
lieu=tx_row["lieu"],
canal=tx_row["canal"]
)

def generate_prompt_resume(client_row, summary_row):
    template = load_template("resume_client.txt")
    return template.format(
client_id=client_row["client_id"],
profession=client_row["profession"],
revenu=client_row["revenu_mensuel"],
nb_transactions=summary_row["nombre_transactions"],
montant_moyen=summary_row["montant_moyen"],
montant_ecart=summary_row["montant_ecart"]
)

def generate_prompt_message_client(tx_row):
    template = load_template("message_client.txt")
    return template.format(
client_id=tx_row["client_id"],
montant=tx_row["montant"],
date=tx_row["date"].strftime("%Y-%m-%d"),
type=tx_row["type"]
)
