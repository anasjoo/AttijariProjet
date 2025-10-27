import pandas as pd

def load_clients(path="../datasets/fake_clients.csv"):
    return pd.read_csv(path)

def load_transactions(path="../datasets/fake_transactions.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
  
    clients_df = load_clients()
    print("Clients Data:")
    print(clients_df.head())

    transactions_df = load_transactions()
    print("\\nTransactions Data:")
    print(transactions_df.head())
