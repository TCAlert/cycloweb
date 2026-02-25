import os
import xarray as xr
import matplotlib.pyplot as plt
import cartopy, cartopy.crs as ccrs
import satcmaps as cmaps
import cmaps as cmp
import image_retrieval as ir
import numpy as np
import cartopy.feature as cfeature
from pyproj import Proj
import scipy
import random
import urllib.request
import ibtracsParser as IP
KEY = 'd605d6cf-11b8-11ec-818a-a0369f818cc4'
OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

def reproject(dataset, lons, lats, res = 0.0179985):                           
    # Extents and interpolation for IR
    minimglat = np.nanmin(lats)
    maximglat = np.nanmax(lats)
    minimglon = np.nanmin(lons)
    maximglon = np.nanmax(lons)
    grid = np.meshgrid(np.arange(minimglat, maximglat, res), np.arange(minimglon, maximglon, res))
    
    lats = lats.flatten()
    lons = lons.flatten()
    data = dataset.flatten()
    
    # Fix shape issue for boolean conditions
    floater_IR = (np.greater(lats, minimglat) & np.greater(maximglat, lats) &
                    np.greater(lons, minimglon) & np.greater(maximglon, lons) & np.isfinite(data))

    gridded_data = scipy.interpolate.griddata((lats[floater_IR], lons[floater_IR]), data[floater_IR], (grid[0], grid[1]), method='cubic')
    
    return gridded_data, grid[1], grid[0]

def mcfetch(satellite, band, date, time, lat, lon, zoom, res = 0.0179985, fd = False):
    if fd == False:
        if (type(zoom) == str) or (int(zoom) == 2):
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=800+800&mag=-1+-1'
        elif int(zoom) == 3:
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=1750+1750&mag=-1+-1'
        else:
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=500+500&mag=-1+-1'
    else:
        if (type(zoom) == str) or (int(zoom) == 2):
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=800+800&coverage=FD&mag=-1+-1'
        elif int(zoom) == 3:
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=1750+1750&coverage=FD&mag=-1+-1'
        else:
            url = f'https://mcfetch.ssec.wisc.edu/cgi-bin/mcfetch?dkey={KEY}&satellite={satellite.upper()}&band={band}&output=NETCDF&date={date}&time={time}&unit=TEMP&lat={lat}+{lon}&size=500+500&coverage=FD&mag=-1+-1'
    print(url)

    try:
        filename = 'mcfetch'
        urllib.request.urlretrieve(url, os.path.join(OUTPUTS, filename + '.nc'))
    except:
        filename = 'mcfetch2'
        urllib.request.urlretrieve(url, os.path.join(OUTPUTS, filename + '.nc'))
    data = xr.open_dataset(os.path.join(OUTPUTS, filename + '.nc'))
    
    data, lons, lats = reproject(data['data'].values, data['lon'].values, data['lat'].values, res)
    
    return data, lons, lats, filename

def map(lon, lat, date, time, zoom = 2, ibtracs = None):
    plt.figure(figsize = (18, 9))

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))
    
    try:
        if int(zoom) == -2:
            ax.set_extent([lon - .5, lon + .5, lat - .5, lat + .5], crs=ccrs.PlateCarree())
        elif int(zoom) == -1:
            ax.set_extent([lon - 1, lon + 1, lat - 1, lat + 1], crs=ccrs.PlateCarree())
        elif int(zoom) == 0:
            ax.set_extent([lon - 2.5, lon + 2.5, lat - 2.5, lat + 2.5], crs=ccrs.PlateCarree())
        elif int(zoom) == 1:
            ax.set_extent([lon - 5, lon + 5, lat - 5, lat + 5], crs=ccrs.PlateCarree())
        elif int(zoom) == 3:
            ax.set_extent([lon - 15, lon + 15, lat - 15, lat + 15], crs=ccrs.PlateCarree())
        else:
            ax.set_extent([lon - 7.5, lon + 7.5, lat - 7.5, lat + 7.5], crs=ccrs.PlateCarree())
    except:
        lat, lon = IP.getCoords(ibtracs, '/'.join(date), time, [zoom, int(date[2])])
        ax.set_extent([lon - 7.5, lon + 7.5, lat - 7.5, lat + 7.5], crs=ccrs.PlateCarree())


    # Add coastlines, borders and gridlines
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth = 0.75)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth = 0.25)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth = 0.25)  
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth = 1, color='gray', alpha=0.5, linestyle='--')   
    gl.top_labels = gl.right_labels = False

def run(satellite, date, TIME, lat, lon, band, cmp = None, zoom = 2, ibtracsCSV = None):
    date = date.split('/')

    try:
        zoom = int(zoom)
    except:
        lat, lon = IP.getCoords(ibtracsCSV, '/'.join(date), TIME, [zoom, int(date[2])])
        zoom = 2

    lon = float(lon)
    lon = str(lon * -1)

    if satellite[0].lower() == 'g' and (int(satellite[1:]) >= 8):
        sattNum = satellite[1:]
        if band == '3':
            ch = '3'
        else:
            ch = '4'
        data, lons, lats, filename = mcfetch(f'GOES{sattNum}', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0179985 * 2)                
        satellite = f'GOES-{sattNum}'
        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"
        res = '4km'
    elif satellite[0].lower() == 'g' and (int(satellite[1:]) < 8):
        sattNum = satellite[1:]
        if band == '3':
            ch = '9'
        else:
            ch = '8'
        data, lons, lats, filename = mcfetch(f'GOES{sattNum}', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0629)                
        if int(sattNum) == 7:
            data = data / 10
        satellite = f'GOES-{sattNum}'
        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"
        res = '4km'
    elif satellite[0].lower() == 's':
        sattNum = -1
        if band == '3':
            ch = '9'
        else:
            ch = '8'
        data, lons, lats, filename = mcfetch(f'{satellite.upper()}', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0629)                
        if int(sattNum) == 7:
            data = data / 10
        satellite = f'{satellite.upper()}'
        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"
        res = '4km'    
    elif int(satellite) >= 8:
        if band == '3':
            ch = '9'
        else:
            ch = '13'

        data, lons, lats, filename = mcfetch(f'HIMAWARI{satellite}', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, fd = True)
        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"

        res = '2km'
        satellite = f'HIMAWARI-{satellite}'
    elif int(satellite) == 1:
        ch = '8'
        
        data, lons, lats, filename = mcfetch(f'GMS1', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0179985 * 2)
        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"
        res = '4km'
        satellite = f'GMS-1'
    else:
        if band == '3':
            ch = '4'
        else:
            ch = '2'

        if int(satellite) == 7:
            data, lons, lats, filename = mcfetch(f'MTSAT2', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0179985 * 2)
            satellite = f'MTSAT-2'
        elif int(satellite) == 6:
            data, lons, lats, filename = mcfetch(f'MTSAT1R', ch, f'{date[2]}{date[0].zfill(2)}{date[1].zfill(2)}', TIME, lat, lon, zoom, res = 0.0179985 * 2)
            satellite = f'MTSAT-1R'

        time = f"{date[2]}-{str(date[0]).zfill(2)}-{str(date[1]).zfill(2)} at {TIME} UTC"
        res = '4km'
    
    if band == '3':
        try:
            if cmp.lower() == 'random':
                rand = random.randrange(0, len(cmaps.wvtables.keys()), 1)
                cmp = list(cmaps.wvtables.keys())[rand]
        
            cmap, vmax, vmin = cmaps.wvtables[cmp.lower()]
        except:
            try:
                cmap, vmax, vmin = cmaps.irtables[cmp]
                vmax = 0
                vmin = -90
            except:
                if cmp == None:
                    cmap, vmax, vmin = cmaps.wvtables['wv']
                else:
                    cmap, vmax, vmin = cmp, 0, -90    
    elif band == '4':
        try:
            if cmp.lower() == 'random':
                rand = random.randrange(0, len(cmaps.irtables.keys()), 1)
                cmp = list(cmaps.irtables.keys())[rand]
        
            cmap, vmax, vmin = cmaps.irtables[cmp.lower()]
        except:
            try:
                cmap, vmax, vmin = cmaps.wvtables[cmp]
                vmax = 40
                vmin = -100
            except:
                if cmp == None:
                    cmap, vmax, vmin = cmaps.irtables['irg']
                else:
                    cmap, vmax, vmin = cmp, 40, -110        

    print(cmp, vmin, vmax)

    map(float(lon) * -1, float(lat), date = date, time = TIME, zoom = zoom, ibtracs = ibtracsCSV)
    
    plt.pcolormesh(lons, lats, data - 273.15, vmin = vmin, vmax = vmax, cmap = cmap)
    cbar = plt.colorbar(orientation = 'vertical', aspect = 50, pad = .02)
    cbar.set_label(cmp.upper())
    plt.title(f'{satellite} Channel {ch.zfill(2)} Brightness Temperature\nTime: {time}' , fontweight='bold', fontsize=10, loc='left')
    plt.title(f'{res}\nDeelan Jariwala', fontsize=10, loc='right')
    plt.savefig(os.path.join(OUTPUTS, 'stormir.png'), dpi=250, bbox_inches='tight')
    
    # plt.show()
    plt.close()

    return filename

# ib = IP.loadCSV()

#run('8', '10/30/2020', '2030', '18', '129.35', '3', 'wv10')
#run('8', '9/13/2016', '1320', '20.57', '122.54', '3', 'wv9')
# run('6', '11/7/2013', '1500', '10.43', '128.01', '4', 'test', 2, ibtracsCSV = ib)
# run('g9', '6/15/2004', '2100', '13.9', '136.7', '4', 'irg', 'Dianmu', ibtracsCSV=ib)
# run('g4', '9/21/1982', '0000', '15.40', '-106.50', '4', 'irg', '2', ibtracsCSV=ib)
# run('g7', '6/15/1993', '1200', '12.1', '-119.2', '4', 'bd4', '1')
# run('g7', '09/13/1988', '2100', '19.5', '-83.6', '4', 'bd3', '2')
# run('sms2', '8/31/1979', '2000', '17.50', '-69.20', '4', 'avn', '2')
