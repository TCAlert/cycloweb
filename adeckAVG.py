import os
import adeck as ad
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['Courier New', 'Liberation Mono', 'DejaVu Sans Mono', 'monospace']
import cartopy, cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature
import cmaps as cmap
import xarray as xr
from datetime import datetime

OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

COLORS = ['#1274c4', '#ff7f00', '#4daf4a', '#f781bf', "#990a16", "#d842ef",  '#2de6f7', '#e41a1c', '#dede00']
USAGE = '```$adeck [storm] [date] [time] [model]```'

def getData(storm, date, time, model):
    date = date.split('/')
    date = f"{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}{str(time).zfill(2)}"
    data = ad.filterData(storm, [date], model, [0, 12, 24, 36, 48, 60, 72, 96, 120], df = False)

    data = data.astype(object)
    data[:, :, 5:10] = data[:, :, 5:10].astype(float)

    newHour = np.nanmean(np.array(data[:, :, 5], dtype = float), axis = 0)
    newLats = np.nanmean(np.array(data[:, :, 6], dtype = float), axis = 0)
    newLons = np.nanmean(np.array(data[:, :, 7], dtype = float), axis = 0)
    newWind = np.nanmean(np.array(data[:, :, 8], dtype = float), axis = 0)

    return data, newLats, newLons, newWind, newHour

def checkIndex(data):
    for x in range(len(data)):
        if np.isnan(data[x]):
            return x
    return 20 

def map(interval, labelsize):
    fig = plt.figure(figsize=(14, 6))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_frame_on(False)
    
    # Add state boundaries to plot
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5, zorder = 4)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth = 0.25, zorder = 3)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth = 0.25, zorder = 2)
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor = 'white', zorder = 1)
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize=labelsize, left = False, bottom = False)
    ax.grid(linestyle = '--', which = 'major', alpha = 0.5, color = 'black', linewidth = 0.5, zorder = 9)

    return ax 

def plot(storm, date, time, models):
    models = [m.upper() for m in models]
    allData, newLats, newLons, newWind, hour = getData(storm, date, time, models)
    hrs = allData[:, :, 5]
    lat = allData[:, :, 6]
    lon = allData[:, :, 7]

    ax = map(5, 9)

    if abs(np.nanmax(lon) - np.nanmin(lon)) < abs(np.nanmax(lat) - np.nanmin(lat)):
        extent = [np.nanmin(lon) - 15, np.nanmax(lon) + 15, np.nanmin(lat) - 5, np.nanmax(lat) + 5]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        extent = [np.nanmin(lon) - 10, np.nanmax(lon) + 10, np.nanmin(lat) - 5, np.nanmax(lat) + 5]
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    file = xr.open_dataset(f"http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{datetime.utcnow().year}.nc")
    latest = file['time'][-1]
    dataset = file['sst'].sel(time = latest, lat = slice(extent[2], extent[3]), lon = slice(360 + extent[0], 360 + extent[1]))

    plt.contour(dataset.lon, dataset.lat, dataset.values, levels = [26, 28, 30], colors = ['#000000', '#000000AA', "#00000070"], linewidths = 1.5, transform=ccrs.PlateCarree(central_longitude=0), zorder = 1)
    plt.pcolormesh(dataset.lon, dataset.lat, dataset.values, vmin = 0, vmax = 32, cmap= cmap.sst(), alpha = 0.5, transform=ccrs.PlateCarree(central_longitude=0), zorder = 0)

    ax.plot(newLons, newLats, linewidth = 2.5, color = '#000000', transform = ccrs.PlateCarree(central_longitude=0), zorder = 14)
    c = ax.scatter(newLons, newLats, c = hour, s = 30, edgecolor = '000000', cmap = cmap.probs2(), transform = ccrs.PlateCarree(central_longitude=0), zorder = 15)
    for x in range(len(models)):
        if models[x] not in models[:x]:
            i = checkIndex(lon[x])
            ax.plot(lon[x][:i], lat[x][:i], linewidth = 1.5, color = COLORS[x % len(COLORS)], transform = ccrs.PlateCarree(central_longitude=0), label = models[x][:i])
            ax.scatter(lon[x][:i], lat[x][:i], c = hrs[x][:i], s = 15, edgecolor = COLORS[x % len(COLORS)], cmap = cmap.probs2(), transform = ccrs.PlateCarree(central_longitude=0), zorder = 10)

    plt.title(f'A-Deck FH00 through FH120 Averaged Consensus\nModels: {", ".join(models)}' , fontweight='bold', fontsize=10, loc='left')
    plt.title(f'Time: {date} at {time}z', fontsize = 10, loc = 'center')
    plt.title(f'{storm.upper()}\nDeelan Jariwala', fontsize=10, loc='right')  
    cbar = plt.colorbar(c, orientation = 'vertical', aspect = 50, pad = .02, label = 'Forecast Hour')
    cbar.ax.set_yticks([int(h) for h in hour[:checkIndex(hour)]])
    plt.legend(loc = 'upper right')

    plt.savefig(os.path.join(OUTPUTS, 'consensusGen.png'), dpi = 200, bbox_inches = 'tight')
    # plt.show()
    plt.close()

    test = np.array([hour, np.round(list(newLats), 1), np.round(list(newLons), 1), np.round(list(newWind), 1)]).T

    return test

# test = plot('al10', '10/10/2025', '06', ['hfai', 'hfbi'])
# print(test)