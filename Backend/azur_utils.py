# Backend/azure_utils.py

import pandas as pd
from azure.storage.blob import BlobServiceClient
import io
import os
import sys

# --- ⚠️ Configuration Azure Blob Storage ⚠️ ---
# Using environment variables (os.environ.get('VAR_NAME')) is STRONGLY recommended for production.
STORAGE_ACCOUNT_NAME = "datasetdetection"
CONTAINER_NAME = "datasets"
# This key is sensitive! For production, use Azure Key Vault or Managed Identity.
STORAGE_ACCOUNT_KEY = "" 
CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

def load_df_from_blob(file_name: str) -> pd.DataFrame:
    """Loads a Pandas DataFrame directly from Azure Blob Storage."""
    
    # Use the connection string for simplicity
    blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    blob_client = blob_service_client.get_container_client(CONTAINER_NAME).get_blob_client(file_name)
    
    try:
        # Download blob content as bytes
        blob_data = blob_client.download_blob().readall()
        
        # Read the bytes into a Pandas DataFrame
        df = pd.read_csv(io.BytesIO(blob_data))
        
        print(f"✅ Successfully loaded {file_name} from Azure Blob Storage.")
        return df

    except Exception as e:
        error_message = f"❌ Error loading {file_name} from Blob Storage. Check account name, container name, and key/connection string: {e}"
        print(error_message)
        # Raise a runtime error to prevent the application from starting with no data
        raise RuntimeError(error_message)

# --- Data Loading (Executed on module import) ---
try:
    clients_df = load_df_from_blob('fake_clients.csv')
    transactions_df = load_df_from_blob('fake_transactions.csv')
except RuntimeError:
    # If loading fails, initialize empty DataFrames to prevent crashes during module import
    clients_df = pd.DataFrame()
    transactions_df = pd.DataFrame()
    print("⚠️ Application starting with empty dataframes due to Blob Storage failure.")

# Merge data for the full DataFrame
full_df = transactions_df.merge(clients_df, on='client_id', how='left')

# Expose the dataframes for import
__all__ = ['clients_df', 'transactions_df', 'full_df']