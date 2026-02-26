import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = ['Courier New', 'Liberation Mono', 'DejaVu Sans Mono', 'monospace']
import cartopy, cartopy.crs as ccrs
import numpy as np
import xarray as xr
import cmaps as cmap
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from matplotlib.offsetbox import AnchoredText
import ibtracsParser as IP
import satcmaps as scmap

OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

USAGE =f'```$primed [sensor: ATMS, AMSU/MHS, AMSR, GMI, SSMI, SSMIS, TMI] [date] [time] [channel: lower, middle] [storm (xxXX format)]```'

sattDICT  = {'mid' :{'atms' : ('S3', 'TB_88.2QV'),
                     'amsu' : ('S1', 'TB_89.0_0.9QV'),
                     'amsr' : ('S5', 'TB_A89.0H'),
                     'gmi'  : ('S1', 'TB_89.0H'),
                     'mhs'  : ('S1', 'TB_89.0V'),
                     'ssmi' : ('S2', 'TB_85.5H'),
                     'ssmis': ('S4', 'TB_91.665H'),
                     'tmi'  : ('S3', 'TB_85.5H')},
             'low' :{'atms' : ('S2', 'TB_31.4QV'),
                     'amsr' : ('S4', 'TB_36.5H'),
                     'gmi'  : ('S1', 'TB_36.64H'),
                     'ssmi' : ('S1', 'TB_37.0H'),
                     'ssmis': ('S2', 'TB_37.0H'),
                     'tmi'  : ('S2', 'TB_37.0H')}
}

def map(interval, labelsize):
    fig = plt.figure(figsize=(18, 9))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_frame_on(False)
    
    # Add state boundaries to plot
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth = 0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth = 0.5)
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize=labelsize, left = False, bottom = False)
    ax.grid(linestyle = '--', alpha = 0.5, color = 'black', linewidth = 0.5, zorder = 9)

    return ax 

def spmap(ax, interval, labelsize, color = 'black'):
    ax.set_frame_on(False)

    # Add state boundaries to plot
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor = color, linewidth = 0.5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor = color, linewidth = 0.5)
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor = color, linewidth = 0.5)
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize=labelsize, left = False, bottom = False)
    ax.grid(linestyle = '--', alpha = 0.5, color = 'black', linewidth = 0.5, zorder = 9)

    return ax 

def getNearestTimes(times, target):
    target = float(target[0:2]) + (float(target[2:4]) / 60)

    minIndex = None
    minValue = float('Inf')
    for x in range(len(times)):
        temp = times[x]
        temp = float(temp[0:2]) + (float(temp[2:4]) / 60) + (float(temp[4:6]) / 3600) 
        if np.abs(temp - target) < minValue:
            minValue = np.abs(temp - target) 
            minIndex = x
    
    return times[minIndex]

def apply_data_range(data, min_val, max_val, min_outbounds="crop", max_outbounds="crop", norm=True, inverse=False):
    if min_outbounds == "crop":
        data = np.clip(data, min_val, max_val)
    if norm:
        data = (data - min_val) / (max_val - min_val)
    if inverse:
        data = 1.0 - data
    return data

def apply_gamma(data, gamma):
    return np.power(data, gamma)

bucket = 'noaa-nesdis-tcprimed-pds'
product_name = 'v01r01'
def getFile(storm, satellite, date, time, bin = 'final'):
    date = date.split('/')
    year = date[2]
    month = str(date[0]).zfill(2)
    day = str(date[1]).zfill(2)
    basin = storm[0:2].upper()
    date = f'{year}{month}{day}'
    dataset = []
    
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = s3_client.get_paginator('list_objects_v2')
    prefix = f'{product_name}/{bin}/{year}/{basin}/{storm[2:]}/'

    response_iterator = paginator.paginate(
        Bucket = bucket,
        Delimiter='/',
        Prefix = prefix,
    )

    files = []
    for page in response_iterator:
        for object in page['Contents']:
            if satellite in object['Key'] and date in object['Key']:
                file = object['Key']
                files.append(file)

    time = getNearestTimes([temp.split('.')[0][-6:] for temp in files], time)
    for x in range(len(files)):
        if f'{date}{time}' in files[x]:
            file = files[x]

    n = 0
    while n < 10:
        try:
            filename = f'tcprimed_data{str(n)}'
            s3_client.download_file(bucket, file, os.path.join(OUTPUTS, filename + ".nc"))
            break
        except:
            n = n + 1

    return time, [temp.split('.')[0][-6:] for temp in files], filename

def plot(storm, satellite, date, time, datatype = 'mid', color = False, ibtracs = None):
    try:
        check = int(storm[2:4])
    except:
        storm = IP.getATCF(ibtracs, date, time, [storm, date.split('/')[2]])
        storm = storm[0:4]

    try:
        time, allTimes, filename = getFile(storm, satellite, date, time, bin = 'final')
    except:
        time, allTimes, filename = getFile(storm, satellite, date, time, bin = 'preliminary')
    num, band = sattDICT[datatype][satellite.lower()]
    
    nc_path = os.path.join(OUTPUTS, filename + ".nc")
    dataset = xr.open_dataset(nc_path, group = f'passive_microwave/{num}', autoclose = True)
    stormData = xr.open_dataset(nc_path, group = f'overpass_storm_metadata/', autoclose = True)
    lat = float(np.nanmean(stormData['storm_latitude'].values))
    lon = float(np.nanmean(stormData['storm_longitude'].values))

    data = dataset[band]
    lats = dataset['latitude']
    lons = dataset['longitude']
    if np.isnan(data.values).all():
        dataset = xr.open_dataset(nc_path, group = f'passive_microwave/S6', autoclose = True)

        band = band.replace('A', 'B')
        data = dataset[band]
        lats = dataset['latitude']
        lons = dataset['longitude']

    ax = map(1, 9)

    ax.set_title(f'{dataset.platform.upper()} {dataset.instrument.upper()} {datatype.capitalize()}-Level Microwave Imagery\nTime: {date} at {time[0:2]}:{time[2:4]} UTC', fontweight='bold', fontsize=10, loc='left')
    ax.set_title(f'{dataset.automated_tropical_cyclone_forecasting_system_storm_identifier.upper()[0:4]}\nDeelan Jariwala', fontsize=10, loc='right') 

    if color == False and datatype == 'mid':
        # cr = ax.pcolormesh(lons, lats, data, vmin = 180, vmax = 291, cmap = cmap.mw().reversed(), transform=ccrs.PlateCarree(central_longitude=0))
        cr = ax.contourf(lons, lats, data, levels = np.arange(180, 290.5, .5), extend = 'both', cmap = cmap.mw().reversed(), transform=ccrs.PlateCarree(central_longitude=0))
        cbar = plt.colorbar(cr, orientation = 'vertical', aspect = 50, pad = .02)
        cbar.ax.tick_params(axis='both', labelsize=8, left = False, bottom = False)
    elif color == False and datatype == 'low':
        # cw = ax.pcolormesh(lons, lats, data, cmap = cmap.llmw(), vmin = 125, vmax = 300, transform=ccrs.PlateCarree(central_longitude=0))
        cw = ax.contourf(lons, lats, data, cmap = cmap.llmw(), levels = np.arange(125, 300.5, .5), extend = 'both', transform=ccrs.PlateCarree(central_longitude=0))
        cbar = plt.colorbar(cw, ax = ax, orientation = 'vertical', aspect = 50, pad = .02)
        cbar.ax.tick_params(axis='both', labelsize=8, left = False, bottom = False)
    elif (datatype == 'mid' and color == True):
        h89 = data
        v89 = dataset[band.replace('H', 'V')]
        red = 1.818 * v89 - 0.818 * h89
        red = apply_data_range(
            red,
            220.0,
            310.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=True
        )
        red = apply_gamma(red, 1.0)

        grn = (h89 - 240.0) / (300.0 - 240.0)
        grn = apply_data_range(
            grn,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False
        )
        grn = apply_gamma(grn, 1.0)

        blu = (v89 - 270.0) / (290.0 - 270.0)
        blu = apply_data_range(
            blu,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False
        )
        blu = apply_gamma(blu, 1.0)
        rgb = np.stack([red, grn, blu], axis=-1)
        ax.pcolormesh(lons, lats, rgb, transform=ccrs.PlateCarree())
    elif (datatype == 'low' and color == True):
        h37 = data
        v37 = dataset[band.replace('H', 'V')]
        red = 2.181 * v37 - 1.181 * h37
        red = apply_data_range(
            red,
            260.0,
            280.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=True,
        )
        red = apply_gamma(red, 1.0)

        grn = (v37 - 180.0) / (300.0 - 180.0)
        grn = apply_data_range(
            grn,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False,
        )
        grn = apply_gamma(grn, 1.0)

        blu = (h37 - 160.0) / (300.0 - 160.0)
        blu = apply_data_range(
            blu,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False,
        )
        blu = apply_gamma(blu, 1.0)
        rgb = np.stack([red, grn, blu], axis=-1)
        ax.pcolormesh(lons, lats, rgb, transform=ccrs.PlateCarree())

    ax.set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())

    at = AnchoredText("Data from TC-PRIMED",
                  prop=dict(size=8, color = 'gray'), frameon=False,
                  loc=4)
    at.patch.set_alpha(.1)
    ax.add_artist(at)

    plt.savefig(os.path.join(OUTPUTS, 'tcprimed_plot.png'), dpi = 400, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    dataset.close()
    stormData.close()

def plot2(storm, satellite, date, time, datatype = 'mid', color = False, ibtracs = None):
    try:
        check = int(storm[2:4])
    except:
        storm = IP.getATCF(ibtracs, date, time, [storm, date.split('/')[2]])
        storm = storm[0:4]

    try:
        time, allTimes, filename = getFile(storm, satellite, date, time, bin = 'final')
    except:
        time, allTimes, filename = getFile(storm, satellite, date, time, bin = 'preliminary')
    num, band = sattDICT[datatype][satellite.lower()]
    print(num, band)
    
    nc_path = os.path.join(OUTPUTS, filename + ".nc")
    dataset = xr.open_dataset(nc_path, group = f'passive_microwave/{num}', autoclose = True)
    stormData = xr.open_dataset(nc_path, group = f'overpass_storm_metadata/', autoclose = True)
    irData = xr.open_dataset(nc_path, group = 'infrared')
    lat = float(np.nanmean(stormData['storm_latitude'].values))
    lon = float(np.nanmean(stormData['storm_longitude'].values))

    ir = irData['IRWIN'] - 273.15

    data = dataset[band]
    lats = dataset['latitude']
    lons = dataset['longitude']
    if np.isnan(data.values).all():
        dataset = xr.open_dataset(nc_path, group = f'passive_microwave/S6', autoclose = True)

        band = band.replace('A', 'B')
        data = dataset[band]
        lats = dataset['latitude']
        lons = dataset['longitude']
    
    if color == True:
        w = -0.4
    else:
        w = -0.25

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 2, wspace = w)

    axes = [0,
            fig.add_subplot(gs[0, 0], projection = ccrs.PlateCarree()),
            fig.add_subplot(gs[0, 1], projection = ccrs.PlateCarree())]
    
    for x in range(1, 3):
        if x == 2:
            axes[x] = spmap(axes[x], 1, 9, color = '#00cccc')
        else:
            axes[x] = spmap(axes[x], 1, 9)

    axes[1].set_title(f'{dataset.platform.upper()} {dataset.instrument.upper()} {datatype.capitalize()}-Level Microwave Imagery\nTime: {date} at {time[0:2]}:{time[2:4]} UTC', fontweight='bold', fontsize=10, loc='left')
    axes[2].set_title(f'{dataset.automated_tropical_cyclone_forecasting_system_storm_identifier.upper()[0:4]}\nDeelan Jariwala', fontsize=10, loc='right') 

    if color == False and datatype == 'mid':
        # cr = ax.pcolormesh(lons, lats, data, vmin = 180, vmax = 291, cmap = cmap.mw().reversed(), transform=ccrs.PlateCarree(central_longitude=0))
        cr = axes[1].contourf(lons, lats, data, levels = np.arange(180, 290.5, .5), extend = 'both', cmap = cmap.mw().reversed(), transform=ccrs.PlateCarree(central_longitude=0))
        cbar = plt.colorbar(cr, ax = axes[1], orientation = 'vertical', aspect = 50, pad = .02)
        cbar.ax.tick_params(axis='both', labelsize=8, left = False, bottom = False)
    elif color == False and datatype == 'low':
        # cw = ax.pcolormesh(lons, lats, data, cmap = cmap.llmw(), vmin = 125, vmax = 300, transform=ccrs.PlateCarree(central_longitude=0))
        cw = axes[1].contourf(lons, lats, data, cmap = cmap.llmw(), levels = np.arange(125, 300.5, .5), extend = 'both', transform=ccrs.PlateCarree(central_longitude=0))
        cbar = plt.colorbar(cw, ax = axes[1], orientation = 'vertical', aspect = 50, pad = .02)
        cbar.ax.tick_params(axis='both', labelsize=8, left = False, bottom = False)
    elif (datatype == 'mid' and color == True):
        h89 = data
        v89 = dataset[band.replace('H', 'V')]
        red = 1.818 * v89 - 0.818 * h89
        red = apply_data_range(
            red,
            220.0,
            310.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=True
        )
        red = apply_gamma(red, 1.0)

        grn = (h89 - 240.0) / (300.0 - 240.0)
        grn = apply_data_range(
            grn,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False
        )
        grn = apply_gamma(grn, 1.0)

        blu = (v89 - 270.0) / (290.0 - 270.0)
        blu = apply_data_range(
            blu,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False
        )
        blu = apply_gamma(blu, 1.0)
        rgb = np.stack([red, grn, blu], axis=-1)
        axes[1].pcolormesh(lons, lats, rgb, transform=ccrs.PlateCarree())
    elif (datatype == 'low' and color == True):
        h37 = data
        v37 = dataset[band.replace('H', 'V')]
        red = 2.181 * v37 - 1.181 * h37
        red = apply_data_range(
            red,
            260.0,
            280.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=True,
        )
        red = apply_gamma(red, 1.0)

        grn = (v37 - 180.0) / (300.0 - 180.0)
        grn = apply_data_range(
            grn,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False,
        )
        grn = apply_gamma(grn, 1.0)

        blu = (h37 - 160.0) / (300.0 - 160.0)
        blu = apply_data_range(
            blu,
            0.0,
            1.0,
            min_outbounds="crop",
            max_outbounds="crop",
            norm=True,
            inverse=False,
        )
        blu = apply_gamma(blu, 1.0)
        rgb = np.stack([red, grn, blu], axis=-1)
        axes[1].pcolormesh(lons, lats, rgb, transform=ccrs.PlateCarree())

    cw = axes[2].pcolormesh(irData['longitude'], irData['latitude'], ir, cmap = scmap.irg()[0], vmin = scmap.irg()[2], vmax = scmap.irg()[1], transform=ccrs.PlateCarree(central_longitude=0))
    cbar = plt.colorbar(cw, ax = axes[2], orientation = 'vertical', aspect = 50, pad = .02)
    cbar.ax.tick_params(axis='both', labelsize=8, left = False, bottom = False)

    axes[1].set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())
    axes[2].set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())

    at = AnchoredText("Data from TC-PRIMED",
                  prop=dict(size=8, color = 'gray'), frameon=False,
                  loc=4)
    at.patch.set_alpha(.1)
    axes[1].add_artist(at)

    plt.savefig(os.path.join(OUTPUTS, 'tcprimed_plot2.png'), dpi = 400, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    dataset.close()
    stormData.close()

# plot2('AL11', 'AMSR', '09/05/2017', '2230', 'low', color = True)
# plot2('EP20', 'AMSR', '10/23/2015', '1200', 'low')
# plot('IO01', 'MHS', '5/18/2020', '0000', 'mid', True)
# plot2('IO06', 'AMSR', '11/14/2007', '1958', 'mid')
# plot2('AL05', 'AMSR', '08/16/2025', '1200', 'mid', color = True)
# plot('Irma', 'AMSR', '09/05/2017', '2230', 'mid', color = True, ibtracs = IP.loadCSV())
