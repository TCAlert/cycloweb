import os
import s3fs
from datetime import datetime
import numpy as np
import urllib.request
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import xarray as xr

OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

# GRIDSAT-GOES Data Retrieval Function
# Sample usage: getData(13, 2010, 9, 13, 1700)
# This retrieves GOES-13 data on 09/13/2010 at 1700z in the GRIDSAT-GOES dataset
def getDataGridsatGOES(satellite, year, month, day, hour):
    print(f'Downloading GRIDSAT-GOES file for {month}/{day}/{year} at {str(hour).zfill(4)}z')
    try:
        # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-GOES data
        filename = f'GridSat-GOES.goes{str(satellite).zfill(2)}.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(4)}.v01.nc'
        url = f'https://www.ncei.noaa.gov/data/gridsat-goes/access/goes/{str(year)}/{str(month).zfill(2)}/{filename}'

        # Download data using urllib as a file called gridsatgoesfile.nc
        try:
            filename = 'gridsatgoesfile'
            urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatgoesfile.nc'))
        except:
            filename = 'gridsatgoesfile2'
            urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatgoesfile2.nc'))
    except:
        # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-GOES data
        filename = f'GridSat-CONUS.goes{str(satellite).zfill(2)}.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(4)}.v01.nc'
        url = f'https://www.ncei.noaa.gov/data/gridsat-goes/access/conus/{str(year)}/{str(month).zfill(2)}/{filename}'

        # Download data using urllib as a file called gridsatgoesfile.nc
        try:
            filename = 'gridsatconusfile'
            urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatconusfile.nc'))
        except:
            filename = 'gridsatconusfile2'
            urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatconusfile2.nc'))
    
    return filename

# GRIDSAT-B1 Data Retrieval Function
# Sample usage: getData(1998, 10, 13, 9)
# This retrieves GRIDSAT-B1 data on 10/13/1998 at 0900z
def getDataGridsatB1(year, month, day, hour):
    print(f'Downloading GRIDSAT-B1 file for {month}/{day}/{year} at {hour}z')
    # Create string variable containing the name of the file, and use this to make a link to the GRIDSAT-B1 data
    filename = f'GRIDSAT-B1.{str(year)}.{str(month).zfill(2)}.{str(day).zfill(2)}.{str(hour).zfill(2)}.v02r01.nc'
    url = f'https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/{str(year)}/{filename}'
    print(url)

    # Download data using urllib as a file called gridsatb1file.nc
    try:
        filename = 'gridsatb1file'
        urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatb1file.nc'))
    except:
        filename = 'gridsatb1file2'
        urllib.request.urlretrieve(url, os.path.join(OUTPUTS, 'gridsatb1file2.nc'))
    
    return filename

# GOES-R Data Retrieval Function
# Sample usage: getDataGOES(16, 2020, 11, 16, '0900', '13')
# This retrieves GOES-16 Channel 13 infrared data on 11/16/2020 at 0900z
def getDataGOES(satellite, year, month, day, time, band):
    # Log into AWS server using anonymous credentials
    fs = s3fs.S3FileSystem(anon=True)
    fs.ls('s3://noaa-goes16/')

    date = datetime(year, month, day)
    days = date.strftime('%j')
    
    # Retrieve files using the given information, add to a numpy array
    files = np.array(fs.ls(f'noaa-goes{satellite}/ABI-L2-CMIPF/{str(year)}/{days}/{time[0:2]}/'))

    # Loop through array in order to find requested band, add to a new list called 'l'
    l = []
    for x in range(len(files)):
        if f'M6C{band.zfill(2)}' in files[x] or f'M3C{band.zfill(2)}' in files[x]:
            l.append(files[x])
    
    # Loop through l in order to find the file with the matching time
    for x in range(len(l)):
        if time in l[x]:
            file = l[x]

    # Download the file, and rename it to goesfile.nc
    try:
        filename = 'goesfile'
        fs.get(file, os.path.join(OUTPUTS, 'goesfile.nc'))
    except:
        filename = 'goesfile2'
        fs.get(file, os.path.join(OUTPUTS, 'goesfile2.nc'))
    print(f'GOES-{satellite} data downloaded for {month}/{day}/{year} at {time}z')

    return filename

# Function that retrieves Himawari-9 tiles of the requested band and melds them together
# Data is returned at full resolution, regardless of band
def getHimawariData(satellite, year, month, day, time, band):
    bucket = f'noaa-himawari{satellite[-1]}'
    product_name = 'AHI-L2-FLDK-ISatSS'
    dat = []
    dataset = []
    band = str(band).zfill(2)
    
    if int(band) > 4:
        res = '020'
    elif int(band) == 3:
        res = '005'
    else:
        res = '010'

    if int(band) in [10, 11, 12, 13, 14, 15]:
        bits = '12'
    elif int(band) == 7:
        bits = '14'
    else:
        bits = '11'
    
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    prefix = f'{product_name}/{year}/{month:02.0f}/{day:02.0f}/{time}/OR_HFD-{res}-B{bits}-M1C{band}'
    print(f'Time: {prefix}')
    kwargs = {'Bucket': bucket,
                'Prefix': prefix}

    resp = s3_client.list_objects_v2(**kwargs)
    files = []
    for x in range(len(resp['Contents'])):
        key = resp['Contents'][x]['Key']
        if key.startswith(prefix):
            files.append(key)

    f = 0
    while f < 10:
        try:
            himawari_dir = os.path.join(OUTPUTS, f'himawari{f}')
            os.makedirs(himawari_dir, exist_ok=True)
            for x in range(len(files)):
                s3_client.download_file(bucket, files[x], os.path.join(himawari_dir, f'tile{x}.nc'))
            break
        except:
            f = f + 1
    f = str(f)

    return f


# Each function here downloads a netCDF file that can easily be opened with packages like xarray or netCDF4. 