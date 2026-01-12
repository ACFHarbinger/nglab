"""
Utilities for downloading datasets from Google Drive.
"""
import requests

def download_file_from_gdrive(file_id, dest):
    """
    Download a file from Google Drive.

    Args:
        file_id (str): Google Drive file ID.
        dest (str): Destination path to save the file.
    """
    def get_confirm_token(response):
        """Extract confirmation token from response cookies."""
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination):
        """Save response content to destination in chunks."""
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    resp = session.get(url=URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response=resp)

    if token:
        params = {'id': file_id, 'confirm': token}
        resp = session.get(url=URL, params=params, stream=True)

    save_response_content(response=resp, destination=dest)