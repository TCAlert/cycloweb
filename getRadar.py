import numpy as np 
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs  # Plot maps
import xarray as xr 
import cartopy.feature as cfeature
import scipy 
import cmaps 
import cartopy.mpl.ticker as cticker
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime 
from file import getGZ

def rePoPolar(dataset):
    r = dataset.range.values
    t = dataset.azimuth.values * (np.pi / 180)
    R, T = np.meshgrid(r, t)
    newX, newY = R * np.cos(T), R * np.sin(T)

    range = np.arange(-3e+05, 3e+05, 300)
    gridX, gridY = np.meshgrid(range, range)
    gridded_data = scipy.interpolate.griddata((newX.flatten(), newY.flatten()), dataset.values.flatten(), (gridX.flatten(), gridY.flatten()), method='nearest')
    gridded_data = gridded_data.reshape(len(range), len(range)).transpose()

    radLat = dataset['latitude'].values
    radLon = dataset['longitude'].values
    lons = radLon + (gridX / (6371000 * np.cos(np.radians(radLat)))) * (180 / np.pi)
    lats = radLat + (gridY / 6371000) * (180 / np.pi)
    
    return lons, lats, gridded_data

def map(interval, labelsize):
    fig = plt.figure(figsize=(18, 9))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_frame_on(False)
    
    # Add state boundaries to plot
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor = 'white', linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor = 'white', linewidth = 0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor = 'white', linewidth = 0.5)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), color = 'black')
    ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'black')
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize=labelsize, left = False, bottom = False)
    ax.grid(linestyle = '--', alpha = 0.5, color = '#545454', linewidth = 0.5, zorder = 9)

    return ax 

def getData(radar):
    bucket = f'unidata-nexrad-level2'    
    
    date = datetime.utcnow()
    time = str(date.hour).zfill(2) + (str(date.minute)).zfill(2)
    year = date.year
    month = str(date.month).zfill(2)
    day = str(date.day).zfill(2)
    hour = time[0:2]
    min = time[2:4]

    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    prefix = f'{year}/{month}/{day}/{radar.upper()}'
    print(prefix)
    kwargs = {'Bucket': bucket,
                'Prefix': prefix}

    resp = s3_client.list_objects_v2(**kwargs)
    files = []
    mins = []
    for x in range(len(resp['Contents'])):
        key = resp['Contents'][x]['Key']
        if key.startswith(f'{prefix}/{radar.upper()}{year}{month}{day}_{hour}'):
            files.append(key)
            mins.append(int(key[31:33]) - int(min))

    f = 0
    while f < 10:
        try:
            s3_client.download_file(bucket, files[np.argmin(np.array(mins)**2)], r"C:\Users\deela\Downloads\radar" + str(f) + ".gz")
            fileName = files[np.argmin(np.array(mins)**2)]
            break
        except:
            f = f + 1
    f = str(f)

    g = 0
    while g < 10:
        try:
            data = xr.open_dataset(r"C:\Users\deela\Downloads\radar" + f + ".gz", engine = 'nexradlevel2', group = f'sweep_{str(g).zfill(2)}')
            break
        except:

            g = g + 1
    
    lons, lats, data = rePoPolar(data['DBZH'])

    return lons, lats, data, fileName

def getPastData(radar, year, month, day, time):
    print(radar, year, month, day, time)
    bucket = f'unidata-nexrad-level2'    
    
    month = str(month).zfill(2)
    day = str(day).zfill(2)
    time = str(time).zfill(4)
    hour = time[0:2]
    min = time[2:4]

    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    prefix = f'{year}/{month}/{day}/{radar.upper()}'
    kwargs = {'Bucket': bucket,
                'Prefix': prefix}

    resp = s3_client.list_objects_v2(**kwargs)
    files = []
    mins = []
    for x in range(len(resp['Contents'])):
        key = resp['Contents'][x]['Key']
        if key.startswith(f'{prefix}/{radar.upper()}{year}{month}{day}_{hour}'):
            files.append(key)
            mins.append(int(key[31:33]) - int(min))

    f = 0
    while f < 10:
        try:
            s3_client.download_file(bucket, files[np.argmin(np.array(mins)**2)], r"C:\Users\deela\Downloads\radar" + str(f) + ".gz")
            fileName = files[np.argmin(np.array(mins)**2)]
            break
        except:
            f = f + 1
    f = str(f)

    g = 0
    while g < 10:
        try:
            data = xr.open_dataset(r"C:\Users\deela\Downloads\radar" + f + ".gz", engine = 'nexradlevel2', group = f'sweep_{str(g).zfill(2)}')
            break
        except:
            try:
                getGZ(r"C:\Users\deela\Downloads\radar" + f + ".gz")
                data = xr.open_dataset(r"C:\Users\deela\Downloads\radar" + f, engine = 'nexradlevel2', group = f'sweep_{str(g).zfill(2)}')
            except:
               g = g + 1
    
    lons, lats, data = rePoPolar(data['DBZH'])

    return lons, lats, data, fileName, f"radar{str(f)}.gz" 

# print(getPastData('klix', '2005', '08', '28', '1800'))