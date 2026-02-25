import numpy as np 
import cartopy, cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import psutil
import os 
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

REGIONS = {'NATL' : ([-100, -10, 0, 50], (16, 6)),
           'TATL' : ([-90, -20, 0, 40], (18, 9)),
           'CATL' : ([-70, -20, 0, 30], (18, 9)),
           'WATL' : ([-100, -50, 2.5, 35], (18, 9)),
           'EATL' : ([-60, 0, -5, 35], (18, 9)),
           'NAFR' : ([-40, 40, 0, 40], (16, 6)),
           'MEDI' : ([-20, 40, 25, 50], (18, 9)),
           'SATL1' : ([-50, -10, -10, -40], (18, 9)),
           'SATL2' : ([-70, -30, -35, -65], (18, 9)),
           'SATL3' : ([-25, 25, 5, -40], (18, 9)),
           'MDR'  : ([-65, -15, 5, 27.5], (16, 6)),
           'CV'  : ([-35, -10, 5, 25], (18, 9)),
           'US'   : ([-130, -60, 20, 60], (18, 9)),
           'WUS'  : ([-140, -100, 25, 57.5], (18, 9)),
           'SAMS' : ([-90, -30, -25, -65], (18, 9)),
           'SAMN' : ([-90, -30, 15, -25], (18, 9)),
           'FNATL' : ([-80, -10, 35, 70], (16, 6)),
           'NWATL': ([-85, -45, 25, 60], (18, 9)),
           'NEATL': ([-50, -10, 25, 60], (18, 9)),
           'SWATL': ([-95, -50, 15, 45], (18, 9)),
           'SEATL': ([-50, -10, 10, 45], (18, 9)),
           'EUS'  : ([-90, -45, 20, 55], (18, 9)),
           'SUB'  : ([-70, -15, 20, 55], (18, 8)),
           'CAG'  : ([-100, -70, 5, 30], (18, 9)),
           'CA'   : ([-120, -60, 0, 40], (18, 9)),
           'CAR'  : ([-90, -55, 5, 26], (18, 9)),
           'GA'   : ([-90, -55, 10, 31], (18, 9)),
           'LA'   : ([-70, -52.5, 7.5, 22.5], (18, 9)),
           'GOM'  : ([-100, -75, 15, 32.5], (18, 9)),
           'EPAC' : ([-140, -80, 0, 30], (16, 6)),
           'EPAC2': ([-160, -100, 0, 30], (16, 6)),
           'CPAC' : ([-179, -119, 0, 30], (16, 6)),
           'HI' :   ([-170, -145, 12.5, 27.5], (18, 9)),
           'NPAC' : ([-189, -99, 20, 70], (24, 8)),
           'NPAC2': ([110, 200, 20, 70], (24, 8)),
           'TPAC' : ([-179, -79, 0, 50], (16, 6)),
           'WPAC' : ([105, 170, 0, 45], (18, 9)),
           'WPAC2' : ([135, 200, 0, 45], (18, 9)),
           'WMDR' : ([110, 160, 5, 27.5], (16, 6)),
           'NMDR' : ([140, 185, -5, 25], (16, 6)),
           'PHIL' : ([105, 140, 5, 26], (16, 6)),
           'NWPAC': ([115, 140, 15, 35], (16, 6)),
           'GUAM' : ([130, 160, 5, 26], (16, 6)),
           'AUS'  : ([100, 165, -45, 0], (18, 9)),
           'SPAC' : ([139, 199, -45, 0], (18, 9)),
           'SCPAC': ([-189, -129, -45, 0], (18, 9)),
           'SEPAC': ([-159, -79, -45, 0], (18, 9)),
           'ENSO' : ([-189, -79, -25, 25], (16, 6)),
           'EQ'   : ([69, 179, -25, 25], (16, 6)),
           'IO'   : ([30, 120, -35, 30], (18, 9)),
           'SWIO' : ([30, 75, 5, -35], (16, 6)),
           'SEIO' : ([75, 120, 5, -35], (16, 6)),
           'SCS'  : ([100, 125, 8, 26], (18, 9)),
           'BOB'  : ([75, 120, 5, 30], (16, 6)),
           'ARB'  : ([35, 80, 5, 30], (16, 6))}

USREGIONS = {'NE' : ([-82.5, -65, 37.5, 48], (18, 9)),
             'MA' : ([-85, -67.5, 33, 42.5], (18, 6)),
             'SE' : ([-95, -77.5, 22.5, 37.5], (18, 9)),
             'SC' : ([-110, -90, 25, 40], (18, 9)),
             'MW' : ([-95, -80, 37.5, 48], (18, 9)),
             'GP' : ([-110, -90, 35, 50], (18, 9)),
             'SW' : ([-130, -105, 30, 42.5], (18, 6)),
             'NW' : ([-130, -110, 40, 50], (18, 9)),
             'WUS': ([-140, -100, 25, 57.5], (18, 9)),
             'EUS': ([-90, -60, 24, 48], (18, 9)),
             'CUS': ([-105, -75, 24, 48], (18, 9)),
             'US' : ([-126, -67, 21, 51], (18, 9))}

RADCONV = np.pi / 180
RADIUSOFEARTH = 3440.1 #nmi

def colored_line(x, y, c, ax=None, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If not provided, the current axes will be used.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn(
            'The provided "array" keyword argument will be overridden',
            UserWarning,
            stacklevel=2,
        )

    xy = np.stack((x, y), axis=-1)
    xy_mid = np.concat(
        (xy[0, :][None, :], (xy[:-1, :] + xy[1:, :]) / 2, xy[-1, :][None, :]), axis=0
    )
    segments = np.stack((xy_mid[:-1, :], xy, xy_mid[1:, :]), axis=-2)
    # Note that
    # segments[0, :, :] is [xy[0, :], xy[0, :], (xy[0, :] + xy[1, :]) / 2]
    # segments[i, :, :] is [(xy[i - 1, :] + xy[i, :]) / 2, xy[i, :],
    #     (xy[i, :] + xy[i + 1, :]) / 2] if i not in {0, len(x) - 1}
    # segments[-1, :, :] is [(xy[-2, :] + xy[-1, :]) / 2, xy[-1, :], xy[-1, :]]

    lc_kwargs["array"] = c
    lc = LineCollection(segments, **lc_kwargs)

    # Plot the line collection to the axes
    ax = ax or plt.gca()
    ax.add_collection(lc)

    return lc

def force_delete(file_path):
    for proc in psutil.process_iter(['pid', 'name']):
        for file in proc.open_files():
            if file.path == file_path:
                print(f"Terminating process {proc.pid} holding {file_path}")
                proc.terminate()
                proc.wait()  # Ensure process is terminated before proceeding


    if os.path.exists(file_path):
        print(file_path)
        try:
            os.remove(file_path)
            print(f"File {file_path} deleted successfully.")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def getV500C(vmax, rmw, latitude):
    x = 0.1147 + 0.0055*vmax - 0.001 * (latitude - 25)

    return vmax * (rmw / 500)**x

def computeR34(vmax, rmw, roci, latitude):
    rmw = rmw * 1.852
    v500c = getV500C(vmax, rmw, latitude)

    return 9 * (v500c * ((roci / 480) + 0.1) + 3)

def density(T, Td, p, q = False):
    Rd = 287.05
    Rv = 461.5
    epsilon = 0.622

    T = T + 273.15
    p = p * 100
    
    # Vapor pressure (Pa)
    if q == True:
        e = (Td * p) / (epsilon + Td * (1 - epsilon))
    else:
        e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5)) * 100
    
    # Partial pressures
    pd = p - e
    
    rho = pd / (Rd * T) + e / (Rv * T)

    return rho

def es(T):
    Lv = 2.5 * 10**6
    Rv = 461.5
    return 611.2 * np.exp(Lv / Rv * (1 / 273.15 - 1 / T))

def moistEnthalpy(t, q):
    cp = 1004
    lv = 2.5 * 10**6

    return cp * t + lv * q

def Gradient2D(data, short = False):
    if short == True:
        lon = 'lon'
        lat = 'lat'
    else:
        lon = 'longitude'
        lat = 'latitude'
    # Define gradient vector as <fx, fy>
    # Compute the derivative of the dataset, A, in x and y directions, accounting for dimensional changes due to centered differencing
    dAx = data.diff(lon)[1:, :]
    dAy = data.diff(lat)[:, 1:]

    # Compute the derivative of both the x and y coordinates
    dx = data[lon].diff(lon) * np.cos(data[lat] * (np.pi / 180)) 
    dy = data[lat].diff(lat)

    # Return dA/dx and dA/dy, where A is the original dataset
    return dAx / dx, dAy / dy

def Gradient2D_m(data, short = False):
    if short == True:
        lon = 'lon'
        lat = 'lat'
    else:
        lon = 'longitude'
        lat = 'latitude'
    # Define gradient vector as <fx, fy>
    # Compute the derivative of the dataset, A, in x and y directions, accounting for dimensional changes due to centered differencing
    dAx = data.diff(lon)[1:, :]
    dAy = data.diff(lat)[:, 1:]

    # Compute the derivative of both the x and y coordinates
    dx = data[lon].diff(lon) * np.cos(np.deg2rad(data[lat])) * (np.pi/180*6_371_000)
    dy = data[lat].diff(lat) * (np.pi/180*6_371_000)

    # Return dA/dx and dA/dy, where A is the original dataset
    return dAx / dx, dAy / dy


def greatCircle(lat1, lon1, lat2, lon2): 
    lat1, lon1, lat2, lon2 = lat1 * RADCONV, lon1 * RADCONV, lat2 * RADCONV, lon2 * RADCONV 
    return RADIUSOFEARTH * np.arccos(np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2) + (np.sin(lat1) * np.sin(lat2)))

def gridlines(ax, interval):
    ax.set_xticks(np.arange(-180, 181, interval), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, interval), crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.tick_params(axis='both', labelsize = interval * 1.5, left = False, bottom = False)
    ax.grid(linestyle = '--', alpha = 0.5, color = 'black', linewidth = 0.5, zorder = 100)

    return ax

def numToMonth(num):
    dict = {1 : 'January',
            2 : 'February',
            3 : 'March',
            4 : 'April',
            5 : 'May',
            6 : 'June',
            7 : 'July',
            8 : 'August',
            9 : 'September',
            10: 'October',
            11: 'November',
            12: 'December'}
    
    return dict[int(num)]

def stripClean(l):
    l = [x.strip() for x in l if x]

    return l 

def strip(l):
    l = [x.strip() for x in l]

    return l 

def steeringLayer(wind):
    if wind < 45:
        top, bot = 700, 850
    elif wind < 60 and wind >= 45:
        top, bot = 500, 850
    elif wind < 90 and wind >= 60:
        top, bot = 400, 850
    elif wind < 112 and wind >= 90:
        top, bot = 300, 850
    elif wind < 122 and wind >= 112:
        top, bot = 250, 850
    else:
        top, bot = 200, 700
    
    return top, bot

def w_comp_of_wind(vvel, temp, pres):
    R = 287
    G = 9.81
    pres = pres * 100

    return -(vvel * R * temp) / (G * pres)

def dptToSph(td, p):
    td = td - 273.15
    e_s = 6.112 * np.exp((17.67 * td) / (td + 243.5))

    epsilon = 0.622
    q = (epsilon * e_s) / (p - (1 - epsilon) * e_s)
    return q

def theta(temp, pres, ref):
    return temp * ((ref / pres) ** (287.052874 / 1005))

def thetae(temp, pres, ref, sh, dew = True):
    t = theta(temp, pres, ref)
    if dew == True:
        dpt = sh 
        sh = dptToSph(dpt, pres)

    return t * (np.e)**((2.501e6 * sh) / (1005 * temp))

def sat_specific_humidity(temp, pres):
    T_C = temp - 273.15

    e_s = 6.112 * np.exp((17.67 * T_C) / (T_C + 243.5))
    eps = 0.622

    q_s = eps * e_s / (pres - (1 - eps) * e_s)
    return q_s

def thetaes(temp, pres, ref):
    ssh = sat_specific_humidity(temp, pres)

    t = theta(temp, pres, ref)
    return t * np.exp((2.501e6 * ssh) / (1005.0 * temp))

def CtoF(temperature):
    return (temperature * (9/5)) + 32

def FtoC(temperature):
    return (temperature - 32) * (5/9)

def dirSpdToUV(direction, magnitude):
    return magnitude * np.cos(np.deg2rad(direction)), magnitude * np.sin(np.deg2rad(direction))