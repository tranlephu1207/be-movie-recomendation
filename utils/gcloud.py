import os
from google.cloud import storage
from google.oauth2 import service_account

# Global variable to store credentials
_credentials = None


def init_credentials(credentials_path: str = None) -> None:
    """
    Initialize Google Cloud credentials from a service account JSON file.
    The credentials will be stored globally and used by other functions.

    Args:
        credentials_path: Path to service account JSON file. If None, will try to get from environment variable.
    """
    global _credentials

    if credentials_path is None:
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        raise ValueError(
            "No credentials path provided and GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        )

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found at: {credentials_path}")

    _credentials = service_account.Credentials.from_service_account_file(
        credentials_path
    )


def get_storage_client() -> storage.Client:
    """
    Get an authenticated Google Cloud Storage client.

    Returns:
        Authenticated storage client

    Raises:
        RuntimeError: If credentials have not been initialized
    """
    global _credentials

    if _credentials is None:
        raise RuntimeError(
            "Google Cloud credentials not initialized. Call init_credentials() first."
        )

    return storage.Client(credentials=_credentials)


def download_blob(
    bucket_name: str, source_blob_name: str, destination_file_name: str
) -> None:
    """
    Downloads a blob from Google Cloud Storage.

    Args:
        bucket_name: Name of the GCS bucket
        source_blob_name: Path to the file in GCS bucket
        destination_file_name: Local path where the file should be downloaded

    Raises:
        RuntimeError: If credentials have not been initialized
    """
    storage_client = get_storage_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    # Download the blob
    blob.download_to_filename(destination_file_name)
