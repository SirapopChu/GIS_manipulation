from osgeo import gdal
import os
import glob
import numpy as np
import shutil
from Google import Create_Service
from googleapiclient.http import MediaFileUpload

# _______________ SYSTEM CONFIG ___________________

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCPOES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCPOES)

input_folder = './data_store'
output_folder = './convert_store'

start_date = '2020-01-01'
end_date = '2020-12-31'

# product_name = ['MCD12Q2', 'MOD13A3', 'MYD13A3', 'MOD14A2', 'MYD14A2', 'MOD28C3', 'MYD28C3']
# product_name = ['MOD13A3', 'MYD13A3', 'MOD14A2', 'MOD28C3', 'MYD28C3']
product_name = ['MOD13A3', 'MCD12Q2', 'MOD14A2']

product_band = {'MCD12Q2': [1, 2, 3],
                'MOD13A3': [1, 2, 3],
                'MYD13A3': [1, 2, 3],
                'MOD14A2': [1],
                'MYD14A2': [1],
                'MOD28C3': [1],
                'MYD28C3': [1],
                'MOD16A3GF': [1, 2, 3, 4],
                'MYD16A3GF': [1, 2, 3, 4]}

"""### Link to reference dataset metadata

https://modis.gsfc.nasa.gov/data/dataprod/
"""
dict_folder_id = {}
def upload_folder_to_drive(folder):
    x = 0
    folder_id = '1anWz6PNvMpVK2lwTQ2ucbXMowo-Tefl0' #folder kepler
    folder_names = [folder]
    mime_types1 = [] # type of file that can search on 'https://kb.hostatom.com/content/20612/'
    for type in folder_names:
        mime_types1.append('application/vnd.google-apps.folder')

    for folder_name, mime_type in zip(folder_names, mime_types1):
        file_metadata = {
            'name': folder_name[16:],
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [folder_id]
        }
        service.files().create(
            body=file_metadata,
        ).execute()

        query = f"parents = '{folder_id}'"
        response = service.files().list(q=query).execute()
        files = response.get('files')

        for band in product_band.get((folder_name[16:].split('_')[0])):
          band = f'band_0{band}'
          product_id = files[x].get('id')
          file_metadata = {
            'name': band,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [product_id]
          }
          service.files().create(
             body=file_metadata,
          ).execute()
          query = f"parents = '{product_id}'"
          response = service.files().list(q=query).execute()
          file = response.get('files')
          band_id = file[0].get('id')
          name_key = files[x].get('name').split('_')[0]+'-'+band
          dict_folder_id.update({name_key:band_id})
        x += 1

def upload_file_to_drive(data, location):
    data = data.split('\\')[3]
    id = dict_folder_id.get(location)
    file_metadata = {
        'name': data,
        'mimeType': 'image/tiff',
        'parents': [id]
    }
    service.files().create(
        body=file_metadata
    ).execute()

def delete_folder(folder):
    try:
        shutil.rmtree(folder)
        print('Folder and its content removed')
    except:
        print('Folder not deleted')

def list_files_by_pattern(folder, pattern):

    # Create the full pattern with folder and pattern
    full_pattern = os.path.join(folder, pattern)

    # Use glob to find files matching the pattern
    matching_files = glob.glob(full_pattern)

    # Return the list of matching files
    return matching_files

product_file_list = {}
for product in product_name:
  product_pattern = f'*{product}*.hdf'
  input_folder = f'{input_folder}/{product}_{start_date}_{end_date}'
  product_file_list[product] = list_files_by_pattern(input_folder, product_pattern)
  input_folder = './data_store'

for product, file_list in product_file_list.items():
  print(f'{product}: {len(file_list)} files')

for product in product_file_list.keys():
  input_folder = f'{input_folder}/{product}_{start_date}_{end_date}'
  product_output_folder = os.path.join(output_folder, f'{product}_{start_date}_{end_date}_tif')
  upload_folder_to_drive(product_output_folder)

  if not os.path.exists(product_output_folder):
    os.makedirs(product_output_folder)

  for file in product_file_list[product]:
    file = file.replace('\\', '/')

    print(f'Reading file {file} __________________ ', end='')

    input_base_file = '.'.join(file.split('/')[-1].split('.')[:-1])

    # Open the HDF4 file
    hdf4 = gdal.Open(file, gdal.GA_ReadOnly)

    # Get subdatasets (MODIS data is stored in subdatasets)
    subdatasets = hdf4.GetSubDatasets()

    print(f'Found {len(subdatasets)} bands in HDF file')

    for band_i in product_band[product]:

      band_output_folder = os.path.join(product_output_folder, f'band_{band_i:02}')

      if not os.path.exists(band_output_folder):
        os.makedirs(band_output_folder)

      output_base_file = f'{input_base_file}_b{band_i:02}.tif'

      output_file = os.path.join(band_output_folder, output_base_file)

      file_in = output_file.split('\\')[1].split('_')[0] + '-' + output_file.split('\\')[2]

      # Replace subdataset_index with the index of the subdataset you need
      modis_dataset = gdal.Open(subdatasets[band_i-1][0], gdal.GA_ReadOnly)

      # Read data from the selected subdataset
      data = modis_dataset.ReadAsArray()
      if len(data.shape) > 2:
        data = data[0]

      # Get geotransform and projection from the subdataset
      geotransform = modis_dataset.GetGeoTransform()
      projection = modis_dataset.GetProjection()

      # Create a new GeoTIFF file
      driver = gdal.GetDriverByName('GTiff')
      out_dataset = driver.Create(output_file, modis_dataset.RasterXSize, modis_dataset.RasterYSize, 1, gdal.GDT_Float32)

      # Set geotransform and projection
      out_dataset.SetGeoTransform(geotransform)
      out_dataset.SetProjection(projection)

      # Write the data to the new file
      out_dataset.GetRasterBand(1).WriteArray(data)

      # Close the datasets
      out_dataset = None
      modis_dataset = None

      print(f'Band {band_i}/{len(product_band[product])} written at {output_file}')
      upload_file_to_drive(output_file, file_in)
  hdf4 = None
  input_folder = './data_store'
  delete_folder(product_output_folder)
  delete_folder(input_folder)

