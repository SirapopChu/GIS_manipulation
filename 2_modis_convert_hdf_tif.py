from osgeo import gdal
import os
import glob
import numpy as np
#import h5py

input_folder = './data_store'
output_folder = './convert_store'

start_date = '2020-01-01'
end_date = '2020-12-31'

# product_name = ['MCD12Q2', 'MOD13A3', 'MYD13A3', 'MOD14A2', 'MYD14A2', 'MOD28C3', 'MYD28C3']
# product_name = ['MOD13A3', 'MYD13A3', 'MOD14A2', 'MOD28C3', 'MYD28C3']
product_name = ['MOD13A3']

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

for product, file_list in product_file_list.items():
  print(f'{product}: {len(file_list)} files')

for product in product_file_list.keys():

  product_output_folder = os.path.join(output_folder, f'{product}_{start_date}_{end_date}_tif')

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

    hdf4 = None