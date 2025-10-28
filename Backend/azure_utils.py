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

# Get the key from the environment variable set during the Docker build or App Service runtime.
# Use a default value (like an empty string) if the variable isn't found to avoid errors 
# during casual local testing without the key.
STORAGE_ACCOUNT_KEY = os.environ.get(
    "BLOB_STORAGE_KEY", 
    "PLACEHOLDER_KEY_FOR_LOCAL_DEV" # This placeholder ensures the app compiles locally
) 

# Only connect if the key is actually present (i.e., not the placeholder)
if STORAGE_ACCOUNT_KEY and STORAGE_ACCOUNT_KEY != "PLACEHOLDER_KEY_FOR_LOCAL_DEV":
    CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT_NAME};AccountKey={STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
else:
    # If the key is missing, set a dummy connection string to prevent immediate crashes
    CONNECTION_STRING = "DUMMY_CONNECTION_STRING"
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