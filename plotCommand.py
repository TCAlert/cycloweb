import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
import satcmaps as cmaps
import cmaps as cmp
import image_retrieval as ir
import numpy as np
import cartopy.feature as cfeature
import pandas as pd
import random
import getRadar as gr
import cartopy.mpl.ticker as cticker
from helper import REGIONS, USREGIONS, greatCircle
import ibtracsParser as IP

OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

def map(lon, lat, date, time, ibtracs = None, zoom = 2, center = 0):
    try:
        zoom = int(zoom)
        plt.figure(figsize = (18, 9))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center))
    
        if zoom == -2:
            ax.set_extent([lon - .5, lon + .5, lat - .5, lat + .5], crs=ccrs.PlateCarree())
        elif zoom == -1:
            ax.set_extent([lon - 1, lon + 1, lat - 1, lat + 1], crs=ccrs.PlateCarree())
        elif zoom == 0:
            ax.set_extent([lon - 2.5, lon + 2.5, lat - 2.5, lat + 2.5], crs=ccrs.PlateCarree())
        elif zoom == 1:
            ax.set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())
        elif zoom == 3:
            ax.set_extent([lon - 15, lon + 15, lat - 15, lat + 15], crs=ccrs.PlateCarree())
        elif zoom == 2:
            ax.set_extent([lon - 7.5, lon + 7.5, lat - 7.5, lat + 7.5], crs=ccrs.PlateCarree())
    except:
        try:
            try:
                extent, size = REGIONS[zoom.upper()]
            except:
                extent, size = USREGIONS[zoom.upper()]
        except:
            lat, lon = IP.getCoords(ibtracs, '/'.join(date), time, [zoom, int(date[2])])
            extent = [lon - 7.5, lon + 7.5, lat - 7.5, lat + 7.5]
            size = (18, 9)

        plt.figure(figsize = size)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=center))

        ax.set_extent(extent, crs = ccrs.PlateCarree())

    # Add coastlines, borders and gridlines
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.75)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth = 0.25)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth = 0.25)  
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color='gray', alpha=0.5, linestyle='--')   
    gl.top_labels = gl.right_labels = False

def map2(interval, labelsize, lon, lat, zoom):
    fig = plt.figure(figsize=(18, 9))

    # Add the map and set the extent
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_frame_on(False)

    # Add state boundaries to plot
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), edgecolor = 'white', linewidth = 0.5, zorder = 100)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor = 'white', linewidth = 0.5, zorder = 100)
    ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor = 'white', linewidth = 0.5, zorder = 100)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), color = 'black')
    ax.add_feature(cfeature.LAND.with_scale('50m'), color = 'black')
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize=labelsize, left = False, bottom = False)
    ax.grid(linestyle = '--', alpha = 0.5, color = '#545454', linewidth = 0.5, zorder = 9)

    if zoom == -2:
        ax.set_extent([lon - .5, lon + .5, lat - .5, lat + .5], crs=ccrs.PlateCarree())
    elif zoom == -1:
        ax.set_extent([lon - 1, lon + 1, lat - 1, lat + 1], crs=ccrs.PlateCarree())
    elif zoom == 0:
        ax.set_extent([lon - 2.5, lon + 2.5, lat - 2.5, lat + 2.5], crs=ccrs.PlateCarree())
    elif zoom == 1:
        ax.set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())

    return ax 

def nexrad(radar, date, time, lat, lon, cmp = 'ref', zoom = 0):
    print(radar, date, time, lat, lon, cmp, zoom)
    date = date.split('/')
    radar = radar.upper()
    lat, lon = float(lat), float(lon)

    csv = pd.read_csv(r"C:\Users\deela\Downloads\nexrad_lat_lon.csv")
    if radar == 'NONE':
        csv['distance'] = greatCircle(lat, lon, csv['Latitude'], csv['Longitude'])
        csv = csv.iloc[csv['distance'].argmin()]
        lons, lats, data, fileName, name = gr.getPastData(csv['Site ID'], date[2], date[0], date[1], time)
        print(fileName)
        site = csv['Site ID']
    else:
        lons, lats, data, fileName, name = gr.getPastData(radar, date[2], date[0], date[1], time)
        csv = csv[csv['Site ID'] == radar.upper()]
        site = radar

    file = fileName.split('/')[-1]
    time = file.split('_')
    time = f'{time[0][8:10]}/{time[0][10:12]}/{time[0][4:8]} at {time[1][0:4]}z'    

    ax = map2(.5, 9, float(lon), float(lat), float(zoom))

    if cmp.lower() == 'random':
        rand = random.randrange(0, len(cmaps.radtables.keys()), 1)
        cmp = list(cmaps.radtables.keys())[rand]

    cmp, vmin, vmax = cmaps.radtables[cmp.lower()]
    data = np.where(data == np.nanmin(data), np.nan, data)

    c = ax.pcolormesh(lons, lats, data, cmap = cmp, vmin = vmin, vmax = vmax)
    plt.colorbar(c, orientation = 'vertical', aspect = 50, pad = .02)
    plt.title(f'{site.upper()} Radar Reflectivity (dBZ)\nTime: {time}', fontweight='bold', fontsize=10, loc='left')
    plt.title(f'Deelan Jariwala', fontsize=10, loc='right')
    plt.savefig(os.path.join(OUTPUTS, 'stormrad.png'), dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()

    return name
#nexrad('tjua', '08/29/2019', '0000', '18', '-65', 'gs', '1')
#nexrad('none', '10/10/2018', '1800', '30.17', '-85.46', 'ref10', '1')
# nexrad('kpah', '12/11/2021', '0329', '36.75', '-88.6', 'spooky', '-2')
#nexrad('ktlx', '5/20/2013', '2030', '35', '-97.5', 'ref3', '-1')
#nexrad('ktlx', '5/1/2008', '2007', '35', '-97.5', 'ref3', '-1')
#nexrad('tjua', '9/18/2022', '2000', '18', '-68.1', 'ref5', '2')
#nexrad('klix', '8/29/2005', '1200', '30', '-90', 'ref6', 2)

def run(satellite, date, TIME, lat, lon, band, cmp = None, zoom = 2, ibtracsCSV = None):
    date = date.split('/')
    if satellite in ['16', '17', '18', '19']:
        if band == '3':
            ch = '9'
        elif band in ['8', '9', '10']:
            ch = band
        elif band in ['11', '12', '13', '14', '15', '16']:
            ch = band
        else:
            ch = '13'
        filename = ir.getDataGOES(satellite, int(date[2]), int(date[0]), int(date[1]), TIME, ch)

        dataset = xr.open_dataset(os.path.join(OUTPUTS, filename + '.nc'), autoclose=True)
        data = dataset['CMI']
        center = dataset['geospatial_lat_lon_extent'].geospatial_lon_center
        time = (dataset.time_coverage_start).split('T')
        time = f"{time[0]} at {time[1][:5]} UTC"

        repo = True
        res = '2km'
        satellite = f'GOES-{satellite}'
    elif satellite[0].lower() == 'h':
        if band == '3':
            ch = '9'
        elif band in ['8', '9', '10']:
            ch = band
        elif band in ['11', '12', '13', '14', '15', '16']:
            ch = band
        else:
            ch = '13'
        filename = ir.getHimawariData(satellite, int(date[2]), int(date[0]), int(date[1]), TIME, ch)
        dataset = xr.open_mfdataset(os.path.join(OUTPUTS, f'himawari{filename}', '*.nc'), autoclose=True)
        data = dataset['Sectorized_CMI']
        center = dataset.product_center_longitude
        time = f"{'/'.join(date)} at {TIME} UTC"

        repo = True
        res = '2km'
        satellite = f'HIMAWARI-{satellite[-1]}'
    elif satellite == 'b1':
        filename = ir.getDataGridsatB1(int(date[2]), int(date[0]), int(date[1]), TIME[0:2])

        dataset = xr.open_dataset(os.path.join(OUTPUTS, filename + '.nc'))
        if int(band) == 4:
            data = dataset['irwin_cdr'].squeeze()
        elif int(band) == 3:
            data = dataset['irwvp'].squeeze()

        time = str(dataset.time.values[0]).split('T')
        time = f"{time[0]} at {time[1][:5]} UTC"

        ch = band
        repo = False
        res = '8km'
        satellite = f'GRIDSAT B1'
    else:
        filename = ir.getDataGridsatGOES(satellite, int(date[2]), int(date[0]), int(date[1]), TIME)
        
        dataset = xr.open_dataset(os.path.join(OUTPUTS, filename + '.nc'))
        data = dataset[f'ch{band}'].squeeze()
        center = dataset['satlon'].values[0]
        time = str(dataset.time.values[0]).split('T')
        time = f"{time[0]} at {time[1][:5]} UTC"

        ch = band
        repo = False
        res = '4km'
        satellite = f'GRIDSAT GOES-{satellite}'
    
    if band in ['3', '8', '9', '10']:
        try:
            if cmp.lower() == 'random':
                rand = random.randrange(0, len(cmaps.wvtables.keys()), 1)
                cmp = list(cmaps.wvtables.keys())[rand]
        
            cmap, vmax, vmin = cmaps.wvtables[cmp.lower()]
        except:
            try:
                cmap, vmax, vmin = cmaps.irtables[cmp.lower()]
                vmax = 0
                vmin = -90
            except:
                if cmp == None:
                    cmap, vmax, vmin = cmaps.wvtables['wv']
                else:
                    cmap, vmax, vmin = cmp, 0, -90    
    elif band in ['4', '11', '12', '13', '14', '15', '16']:
        try:
            if cmp.lower() == 'random':
                rand = random.randrange(0, len(cmaps.irtables.keys()), 1)
                cmp = list(cmaps.irtables.keys())[rand]
        
            cmap, vmax, vmin = cmaps.irtables[cmp.lower()]
        except:
            try:
                cmap, vmax, vmin = cmaps.wvtables[cmp.lower()]
                vmax = 40
                vmin = -100
            except:
                if cmp == None:
                    cmap, vmax, vmin = cmaps.irtables['irg']
                else:
                    cmap, vmax, vmin = cmp, 40, -110        

    print(cmp, vmin, vmax)

    try:
        if zoom.lower() in ['spac', 'scpac', 'enso', 'npac', 'npac2', 'nmdr', 'wpac2'] or (float(lon) < 196 and float(lon) > 164) or (float(lon) < -164 and float(lon) > -196):
            map(float(lon), float(lat), date = date, time = TIME, zoom = zoom, center = 180, ibtracs = ibtracsCSV)
        else:
            map(float(lon), float(lat), date = date, time = TIME, zoom = zoom, ibtracs = ibtracsCSV)
    except:
        map(float(lon), float(lat), date = date, time = TIME, zoom = zoom, ibtracs = ibtracsCSV)

    if repo == True:
        plt.imshow(data - 273, origin = 'upper', transform = ccrs.Geostationary(central_longitude = center, satellite_height=35786023.0), vmin = vmin, vmax = vmax, cmap = cmap)
    else:
        if zoom.lower() in ['spac', 'scpac', 'enso', 'npac', 'npac2', 'nmdr', 'wpac2'] or (float(lon) < 190 and float(lon) > 170) or (float(lon) < -164 and float(lon) > -196):
            print(lon)
            plt.pcolormesh(data.lon, data.lat, data.values - 273, vmin = vmin, vmax = vmax, cmap = cmap, transform = ccrs.PlateCarree(central_longitude = 0))
        else:
            plt.pcolormesh(data.lon, data.lat, data.values - 273, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar = plt.colorbar(orientation = 'vertical', aspect = 50, pad = .02)
    cbar.set_label(cmp.upper())
    #cbar.ax.set_yticks(np.arange(170, 370, 40))
    plt.title(f'{satellite} Channel {ch.zfill(2)} Brightness Temperature\nTime: {time}' , fontweight='bold', fontsize=10, loc='left')
    plt.title(f'{res}\nDeelan Jariwala', fontsize=10, loc='right')
    plt.savefig(os.path.join(OUTPUTS, 'stormir.png'), dpi=250, bbox_inches='tight')
    # plt.show()
    plt.close()
    dataset.close()

    return filename

band = '3'
cmap = 'wv53'

# run('16', '10/29/2018', '1800', '17', '-60', band, cmap, 'sub')
# run('16', '12/20/2024', '0600', '17', '-60', band, cmap, region)
# run('16', '10/07/2024', "1930", "0", '0', band, cmap, 'gom')
# run('16', '11/15/2020', '0600', '13', '-77.1', band, cmap, 'car')
# run('16', '11/15/2020', '1200', '13', '-77.1', band, cmap, 'car')
# run('16', '10/28/2020', '2100', '17', '-60', band, cmap, 'us')
# run('16', '9/5/2017', '2230', '17', '-60', band, cmap, 'mdr')
# run('16', '9/8/2017', '1800', '17', '-60', band, cmap, 'watl')#, ibtracsCSV = ib)
# run('16', '9/9/2017', '0000', '22.1', '-77.2', band, cmap, 'ga')
# run('16', '1/4/2018', '1500', '16.1', '-113.7', band, cmap, 'nwatl')
# run('18', '9/7/2023', '0600', '16.1', '-113.7', band, cmap, 'epac')
#run('9', '4/21/1998', '0000', '30', '-80', band, cmap, region)
#run('b1', '5/11/1983', '1200', '18.1', '125', band, cmap, region)
#run('b1', '3/13/1993', '1800', '18.1', '125', band, cmap, region)
# run(11, '12/19/2010', "0900", "21", '179', band, cmap, region)
#run('12', '9/16/2004', "0700", "16.68", "-82.77", band, cmap, region)
#run("8", '9/13/1999', "1800", '24.3', '-73.1', "4", band, cmap, region)
# run('16', "9/9/2024", "1230", "0", "0", band, cmap, 'gom')
# run('16', "9/27/2024", "0000", "0", "0", band, cmap, 'sub')
# run('16', '09/28/2024', "0500", '0', '0', band, cmap, 'sub')
# run('h9', '5/26/2023', '0000', '0', '0', band, cmap, 'wmdr')
# run('h9', '10/11/2023', '0840', '18.65', '143.1', band, cmap, 'guam')
# run('h9', "11/10/2024", '0000', '0', '0', band, cmap, 'phil')
# run('16', '10/21/2020', '2320', '29.56', '-60.29', band, cmap, '3')
# run('16', '9/23/2022', '1800', '0', '0', band, cmap, 'nwatl')
# run('h8', '04/15/2021', '1800', '0', '0', band, cmap, 'phil')
# run('h8', '04/17/2021', '1300', '0', '0', band, cmap, 'phil')
# run('16', '9/6/2017', '2030', '18.8', '-65.2', band, cmap, 'ga')
# run('15', '8/29/2015', '1800', '0', '0', band, cmap, 'tpac')
# run('19', '8/10/2025', '2030', '0', '0', band, cmap, 'cv')
# run('16', '09/25/2022', '2230', '16.625', '-80.4', band, cmap, '2')
# run('19', '9/27/2025', '2130', '0', '0', band, cmap, 'swatl')
# run('19', '10/28/2025', '1410', '17.6', '-78.07', band, cmap, '1')
# run('19', '10/28/2025', '1240', '17.6', '-78.07', band, cmap, '3')
# run('19', '10/28/2025', '0300', '17.6', '-78.07', band, cmap, '2')
# run('13', '1/14/2016', '1500', '17.6', '-78.07', band, cmap, 'natl')