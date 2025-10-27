def compute_client_summary(df_transactions):
    df_transactions["mois"] = df_transactions["date"].dt.to_period("M")
    grouped = df_transactions.groupby("client_id").agg({
        "montant": ["mean", "std", "count"]
    })
    grouped.columns = ["montant_moyen", "montant_ecart", "nombre_transactions"]
    return grouped.reset_index()
