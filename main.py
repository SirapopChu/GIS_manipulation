# from osgeo import gdal
# print(gdal.__version__)
from osgeo import gdal
import os
import glob
import numpy as np
import shutil
from Google import Create_Service
from googleapiclient.http import MediaFileUpload
CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCPOES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCPOES)
folder_id = '1p0BICZpj2KO7rpjKieGOUcGdzJ9Igeev'

query = f"parents = '{folder_id}'"
response = service.files().list(q=query).execute()
file = response.get('file')
print(file)
