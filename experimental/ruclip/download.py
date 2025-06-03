import os
import requests
import tarfile

URL = 'https://files.pythonhosted.org/packages/6f/e5/8e09d95e944d46eabdaafea6aab494e311edece1c5f5631e212676c3ca5b/ruclip-0.0.2.tar.gz'
ARCHIVE_NAME = 'ruclip-0.0.2.tar.gz'

with requests.get(URL, stream=True) as r:
    r.raise_for_status()
    with open(ARCHIVE_NAME, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

with tarfile.open(ARCHIVE_NAME, 'r:gz') as tar:
    tar.extractall()

os.remove(ARCHIVE_NAME)