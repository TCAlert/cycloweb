import os
import asyncio
import traceback
import tempfile
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Output directory — all generated files land here
# ---------------------------------------------------------------------------
OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', os.path.join(tempfile.gettempdir(), 'cyclobot_outputs'))
os.makedirs(OUTPUTS, exist_ok=True)
os.environ['CYCLOBOT_OUTPUTS'] = OUTPUTS  # propagate to the plotting modules

# Data directory for static large files (pickle, NC)
DATA_DIR = os.environ.get('CYCLOBOT_DATA', os.path.dirname(os.path.abspath(__file__)))
os.environ['CYCLOBOT_DATA'] = DATA_DIR

import plotCommand as plot
import mcfetching as mcfetch
import hafs as hafs_module
import TCPRIMEDRetrieve as tcprimed_module
import adeckAVG as adeck_module
import shipsRI as ships_module
import hurdatTrackDensityMaps as hurdat_module

# IBTrACS not loaded on the web — storm-name zoom is not supported here
ibtracsCSV = None

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title='CycloBot API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],   # restrict to your GitHub Pages URL in production
    allow_methods=['GET'],
    allow_headers=['*'],
)

# Serve generated output files as static assets
app.mount('/outputs', StaticFiles(directory=OUTPUTS), name='outputs')


async def run_blocking(func, *args, **kwargs):
    """Run a blocking function in a thread pool so the event loop stays free."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/plot', summary='Satellite IR/WV plot (GOES / Himawari)')
async def plot_endpoint(
    satellite: str = Query(..., description='Satellite name, e.g. G16, G18, H9'),
    date:      str = Query(..., description='Date in MM/DD/YYYY format'),
    time:      str = Query(..., description='Time as 4-digit UTC, e.g. 1800'),
    lat:       str = Query(..., description='Center latitude'),
    lon:       str = Query(..., description='Center longitude'),
    band:      str = Query(..., description='Band/channel number'),
    cmap:      Optional[str] = Query(None, description='Colormap name (optional)'),
    zoom:      Optional[str] = Query(None, description='Zoom level (optional)'),
):
    try:
        extra = [a for a in [cmap, zoom] if a is not None]
        filename = await run_blocking(
            plot.run, satellite, date, time, lat, lon, band, *extra,
            ibtracsCSV=ibtracsCSV
        )
        return {
            'image_url': f'/outputs/stormir.png',
            'nc_url':    f'/outputs/{filename}.nc',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())



@app.get('/mcfetch', summary='McIDAS satellite fetch plot')
async def mcfetch_endpoint(
    satellite: str = Query(..., description='Satellite name'),
    date:      str = Query(..., description='Date in MM/DD/YYYY format'),
    time:      str = Query(..., description='Time as 4-digit UTC, e.g. 1800'),
    lat:       str = Query(..., description='Center latitude'),
    lon:       str = Query(..., description='Center longitude'),
    band:      str = Query(..., description='Band/channel number'),
    cmap:      Optional[str] = Query(None, description='Colormap name (optional)'),
    zoom:      Optional[str] = Query(None, description='Zoom level 1-2 (optional)'),
):
    try:
        # Cap zoom at 2 for all web users
        if zoom is not None:
            try:
                zoom = str(min(int(zoom), 2))
            except ValueError:
                pass
        extra = [a for a in [cmap, zoom] if a is not None]
        filename = await run_blocking(
            mcfetch.run, satellite, date, time, lat, lon, band, *extra,
            ibtracsCSV=ibtracsCSV
        )
        return {
            'image_url': f'/outputs/stormir.png',
            'nc_url':    f'/outputs/{filename}.nc',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/hafs', summary='HAFS model diagnostic plot')
async def hafs_endpoint(
    storm: str = Query(..., description='Storm ID, e.g. al092024'),
    date:  str = Query(..., description='Init date in MM/DD/YYYY format'),
    init:  str = Query(..., description='Init hour as 2-digit UTC, e.g. 00, 06, 12, 18'),
    fhour: str = Query(..., description='Forecast hour, e.g. 024'),
    var:   str = Query(..., description='Variable: temp, dewp, wind, vvel, rh, vort, tadv, mfc, div, pwat, cape, cinh, sst, gust, slp, eff, mpi, diseq, shear, ref'),
    level:  Optional[str] = Query(None, description='Pressure level in hPa (optional)'),
    model:  Optional[str] = Query(None, description='Model variant: a or b (optional, default a)'),
    domain: Optional[str] = Query(None, description='Domain type: storm or synoptic (optional, default storm)'),
):
    try:
        kwargs = {}
        if level is not None:
            kwargs['level'] = level
        if model is not None:
            kwargs['model'] = model
        if domain is not None:
            kwargs['t'] = domain
        await run_blocking(
            hafs_module.hafsPlot, storm.lower(), date, init, fhour, var, **kwargs
        )
        return {
            'image_url': '/outputs/hafsRealTime.png',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/tcprimed', summary='TC-PRIMED passive microwave + IR plot')
async def tcprimed_endpoint(
    storm:    str = Query(..., description='Storm ID in xxXX format, e.g. AL11, EP20, IO06'),
    satellite: str = Query(..., description='Sensor: ATMS, AMSR, GMI, SSMI, SSMIS, MHS, TMI'),
    date:     str = Query(..., description='Date in MM/DD/YYYY format'),
    time:     str = Query(..., description='Time as 4-digit UTC, e.g. 1800'),
    datatype: Optional[str] = Query(None, description='Channel level: mid (85-91 GHz) or low (36-37 GHz), default mid'),
    color:    Optional[str] = Query(None, description='RGB composite: true or false, default false'),
):
    try:
        kwargs = {}
        if datatype is not None:
            kwargs['datatype'] = datatype
        color_bool = str(color).lower() == 'true' if color is not None else False
        dt = (datatype or 'mid').lower()
        await run_blocking(
            tcprimed_module.plot2, storm.upper(), satellite.upper(), date, time,
            dt, color_bool
        )
        return {
            'image_url': '/outputs/tcprimed_plot2.png',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/adeckavg', summary='A-Deck averaged consensus track')
async def adeckavg_endpoint(
    storm:  str = Query(..., description='ATCF storm ID, e.g. al10'),
    date:   str = Query(..., description='Init date in MM/DD/YYYY format'),
    time:   str = Query(..., description='Init hour as 2-digit UTC, e.g. 06'),
    models: str = Query(..., description='Comma-separated model list, e.g. hfai,hfbi,hwrf'),
):
    try:
        model_list = [m.strip() for m in models.split(',') if m.strip()]
        table = await run_blocking(
            adeck_module.plot, storm.lower(), date, time, model_list
        )
        rows = [[round(float(r[0])), round(float(r[1]), 1), round(float(r[2]), 1), round(float(r[3]), 1)]
                for r in table if not any(str(v) == 'nan' for v in r)]
        return {
            'image_url': '/outputs/consensusGen.png',
            'table': rows,
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/shipsri', summary='SHIPS-based random forest RI prediction')
async def shipsri_endpoint(
    storm: str = Query(..., description='Storm ATCF ID, e.g. al13'),
    year:  str = Query(..., description='4-digit year, e.g. 2025'),
    month: str = Query(..., description='Month number, e.g. 10'),
    day:   str = Query(..., description='Day number, e.g. 24'),
    hour:  Optional[str] = Query(None, description='UTC hour: 00 06 12 18 (optional, tries all if omitted)'),
):
    try:
        await run_blocking(
            ships_module.getShips, storm.lower(), year, month, day, hour
        )
        return {
            'image_url': '/outputs/SHART.png',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/hurdatdensity', summary='HURDAT2 track/ACE/RI density anomaly')
async def hurdatdensity_endpoint(
    years:    str = Query(..., description='Comma-separated year list, e.g. 2020,2021'),
    datatype: Optional[str] = Query(None, description='track / RI / 24hrchange / ace / wind (default track)'),
):
    try:
        year_list = [int(y.strip()) for y in years.split(',') if y.strip()]
        dt = (datatype or 'track').lower()
        await run_blocking(hurdat_module.makePlot, year_list, dt)
        return {
            'image_url': '/outputs/hurdatDensityPlot.png',
        }
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get('/health')
async def health():
    return {'status': 'ok'}
