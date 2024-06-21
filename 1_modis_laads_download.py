import requests
import json
import os
import shutil
import pandas as pd
from datetime import datetime, timedelta
from config import TOKEN_KEY
from Google import Create_Service
from googleapiclient.http import MediaFileUpload

CLIENT_SECRET_FILE = 'client_secret.json'
API_NAME = 'drive'
API_VERSION = 'v3'
SCPOES = ['https://www.googleapis.com/auth/drive']

service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCPOES)

thailand_tiles = ['h27v07', 'h27v08','h28v07', 'h28v08']

start_date = '2011-01-01'
end_date = '2020-12-31'

product_name = ['MCD12Q2']#, 'MOD13A3']#, 'MYD13A3', 'MOD14A2', 'MYD14A2', 'MOD16A3', 'MYD16A3'] # see product name at https://modis.gsfc.nasa.gov/data/dataprod/

token = TOKEN_KEY




# _______________ SYSTEM CONFIG ___________________

base_url = 'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/'


def filter_by_hv_patterns(data, hv_patterns):
    """
    Filters the list of dictionaries based on specified h##v## patterns in 'downloadsLink'.

    Args:
    data (list): A list of dictionaries, each containing a 'downloadsLink' key with URLs.
    hv_patterns (list): A list of h##v## patterns to filter by.

    Returns:
    list: A filtered list of dictionaries where 'downloadsLink' contains any of the h##v## patterns.
    """
    filtered_data = []

    for item in data:
        # Extracting h##v## pattern from 'downloadsLink'
        for pattern in hv_patterns:
            if pattern in item['downloadsLink']:
                filtered_data.append(item)
                break  # Once matched, no need to check other patterns for the same item

    if len(filtered_data) == 0:
      filtered_data.extend(data)

    return filtered_data



def download_files(data, download_folder):
    """
    Downloads files from the URLs specified in the 'downloadsLink' of each dictionary in the list.

    Args:
    data (list): A list of dictionaries, each containing a 'downloadsLink' key with URLs.
    download_folder (str): The folder where files should be saved.
    """
 
    headers = {"Authorization": f'Bearer {token}'}

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)  # Create the folder if it doesn't exist

    for item in data:
        url = item['downloadsLink']
        file_name = url.split('/')[-1]  # Extract the file name from the URL
        file_path = os.path.join(download_folder, file_name)
        # Download and save the file
        for attempt in range(3):

          try:
              response = requests.get(url, stream=True, headers=headers)
              response.raise_for_status()  # Check for HTTP request errors

              with open(file_path, 'wb') as file:
                  for chunk in response.iter_content(chunk_size=8192):
                      file.write(chunk)
              print(f"Downloaded {file_name} to {download_folder}")

          except requests.exceptions.HTTPError as err:
              print(f"HTTP Error: {err}")

          except Exception as e:
              print(f"Error downloading {file_name}: {e}")

          else: break



def doy_to_date(year, doy):
    """Convert Day of Year to a regular date."""
    return datetime(year, 1, 1) + timedelta(doy - 1)

def filter_by_date(data, start_date, end_date):
    """
    Filters the data based on a date range.

    :param data: List of data items with DOY dates.
    :param start_date: Start date in 'YYYY-MM-DD' format.
    :param end_date: End date in 'YYYY-MM-DD' format.
    :return: Filtered data.
    """
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    filtered_data = []
    for item in data:
        #print(item)
        year, doy = map(int, item['self'].split('/')[-2:])
        #print(year, doy)
        item_date = doy_to_date(year, doy)
        #print(item_date)
        if start_date <= item_date <= end_date:
            filtered_data.append(item)

    return filtered_data

def extract_year_and_period(start_date, end_date):
    # Convert string dates to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Initialize the output dictionary
    output = {}

    # Current year to start from
    current_year = start.year

    while current_year <= end.year:
        # Start of the year
        year_start = datetime(current_year, 1, 1)

        # End of the year
        year_end = datetime(current_year, 12, 31)

        # Adjusting the start and end dates based on the input range
        if current_year == start.year:
            period_start = start
        else:
            period_start = year_start

        if current_year == end.year:
            period_end = end
        else:
            period_end = year_end

        # Adding to the output dictionary
        output[str(current_year)] = [period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d")]

        # Move to the next year
        current_year += 1

    return output

def upload_to_drive(file_name_list, folder):
    folder_id = '1anWz6PNvMpVK2lwTQ2ucbXMowo-Tefl0' #folder kepler
    folder_names = [folder]
    mime_types1 = [] # type of file that can search on 'https://kb.hostatom.com/content/20612/'
    for type in folder_names:
        mime_types1.append('application/vnd.google-apps.folder')

    for folder_name, mime_type in zip(folder_names, mime_types1):
        file_metadata = {
            'name': folder_name[2:],
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [folder_id]
        }

        service.files().create(
            body=file_metadata,
        ).execute()

    query = f"parents = '{folder_id}'"
    response = service.files().list(q=query).execute()
    files = response.get('files')
    
    file_names = []
    mime_types = [] # type of file that have search on 'Commonly Used mime types'

    for item in file_name_list:
        url = item['downloadsLink']
        file_name = url.split('/')[-1]  # Extract the file name from the URL
        file_names.append(file_name)

    for type in file_names:
        if type[-3:] == 'hdf':
            mime_types.append('application/x-hdf')

    # print(folder)
    # print(file_names)
    # print(mime_types)
    # print(f'{folder}/{file_name}')
    for file_name, mime_type in zip(file_names, mime_types):
        for x in range(len(files)):
            if files[x].get('name').split('_')[0] == file_name.split('.')[0]:
                id = files[x].get('id')
                file_metadata = {
                    'name': file_name,
                    'parents': [id] 
                }
            else:
                pass

        media = MediaFileUpload(f'{folder}/{file_name}', mimetype=mime_type)

        service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

def delete_folder(folder):
    try:
        shutil.rmtree(folder)
        print('Folder and its content removed')
    except:
        print('Folder not deleted')

year_period = extract_year_and_period(start_date, end_date)
print(year_period)


#sat's edited
for product in product_name:

  download_list = []
  save_to_folder = f'./{product}_{start_date}_{end_date}'

  for year in year_period.keys():

          query_by_year = requests.get(f'https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{year}.json').json()['content']
          #query_by_year = json.loads(query_by_year.content)['content']

          query_by_year = filter_by_date(query_by_year, year_period[year][0], year_period[year][1])

          for doy_item in query_by_year:

              print(doy_item['name'])
              query_by_year_doy = requests.get(f'''https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/{product}/{year}/{doy_item['name']}.json''')
              query_by_year_doy = json.loads(query_by_year_doy.content)['content']

              query_by_year_doy_hv = filter_by_hv_patterns(query_by_year_doy, thailand_tiles)

              download_list.extend(query_by_year_doy_hv)

  download_files(download_list[:], save_to_folder)
  #upload_to_drive(download_list[:], save_to_folder)
  #delete_folder(save_to_folder)