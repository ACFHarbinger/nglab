import requests

def download_file_from_gdrive(file_id, dest):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination):
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