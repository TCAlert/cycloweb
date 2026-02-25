from datetime import datetime
import pandas as pd
import urllib.request as urllib
import ssl
import pandas as pd
import json 
ssl._create_default_https_context = ssl._create_unverified_context

def getATCF(parse = False):
    link = 'https://science.nrlmry.navy.mil/geoips/tcdat/sectors/sector_file'#'https://www.nrlmry.navy.mil/tcdat/sectors/atcf_sector_file'     
    data = (urllib.urlopen(link).read().decode('utf-8')).strip()
    if parse == False:
        return data
    else:
        data = data.split('\n')
        for x in range(len(data)):
            data[x] = (data[x].split(' '))
            for y in range(len(data[x])):
                data[x][y] = data[x][y].strip()
        
        df = pd.DataFrame(data)
        for x in range(len(df[0])):
            df[2][x] = f'20{df[2][x][0:2]}-{df[2][x][2:4]}-{df[2][x][4:6]} at {df[3][x]}z' 
            if df[4][x][-1] == 'S':
                df[4][x] = float(df[4][x][:-1]) * -1
            else:
                df[4][x] = float(df[4][x][:-1])

            if df[5][x][-1] == 'W':
                df[5][x] = float(df[5][x][:-1]) * -1
            else:
                df[5][x] = float(df[5][x][:-1])
        df = df.drop(3, axis = 1)
        
        id, time, lat, lon = df[0], df[2], df[4], df[5]
        return id, time, lat, lon
            
# Retrieve most recent fix in the NHC or JTWC Best Track for a given storm
def mostRecent(storm):
    year = str(datetime.now().year)
    if ('al' in storm.lower() or 'ep' in storm.lower() or 'cp' in storm.lower()):
        try:
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/NHC/b' + storm.lower() + year + '.dat' 
            # link = 'https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/NHC/b' + storm.lower() + year + '.dat'     
            data = urllib.urlopen(link).read().decode('utf-8')  
        except:      
            try:
                link = 'https://ftp.nhc.noaa.gov/atcf/btk/b' + storm.lower() + year + '.dat'  
                data = urllib.urlopen(link).read().decode('utf-8')     
            except:
                link = 'https://www.emc.ncep.noaa.gov/gc_wmb/vxt/DECKS/b' + storm.lower() + year + '.dat'  
                data = urllib.urlopen(link).read().decode('utf-8')     

    else:
        if ('sh' in storm.lower() and datetime.now().month >= 11):
            year = str(int(year) + 1)
        try:
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/JTWC/b' + storm.lower() + year + '.dat' 
            # link = 'https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/b' + storm.lower() + year + '.dat'    
            data = urllib.urlopen(link).read().decode('utf-8')           
        except:
            try:
                link = f'https://www.nrlmry.navy.mil/atcf_web/docs/tracks/{year}/b{storm.lower()}{year}.dat'          
                data = urllib.urlopen(link).read().decode('utf-8')     
            except:
                try:
                    link = 'https://www.emc.ncep.noaa.gov/gc_wmb/vxt/DECKS/b' + storm.lower() + year + '.dat'  
                    data = urllib.urlopen(link).read().decode('utf-8')     
                except:
                    link = f'https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{storm.lower()}{year}.dat' 
                    data = urllib.urlopen(link).read().decode('utf-8')     

    line = data.split("\n")
    return line[-2]         

# Retrieve best track data for a given TC in a Pandas Dataframe
def getStorm(storm):
    year = str(datetime.now().year)
    if ('al' in storm.lower() or 'ep' in storm.lower() or 'cp' in storm.lower()):
        try:
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/NHC/b' + storm.lower() + year + '.dat' 
            # link = 'https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/NHC/b' + storm.lower() + year + '.dat' 
            data = pd.read_csv(link, header = None, usecols=range(20))
        except:
            try:
                link = 'https://ftp.nhc.noaa.gov/atcf/btk/b' + storm.lower() + year + '.dat'     
                data = pd.read_csv(link, header = None, usecols=range(20))          
            except:
                link = 'https://www.emc.ncep.noaa.gov/gc_wmb/vxt/DECKS/b' + storm.lower() + year + '.dat'  
                data = pd.read_csv(link, header = None, usecols=range(20))  
    else:
        if ('sh' in storm.lower() and datetime.now().month >= 11):
            year = str(int(year) + 1)
        try:
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/JTWC/b' + storm.lower() + year + '.dat' 
            # link = 'https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/b' + storm.lower() + year + '.dat'          
            data = pd.read_csv(link, header = None, usecols=range(20))
        except:
            try:
                link = f'https://www.nrlmry.navy.mil/atcf_web/docs/tracks/{year}/b{storm.lower()}{year}.dat'          
                data = pd.read_csv(link, header = None, usecols=range(20))
            except:
                link = f'https://hurricanes.ral.ucar.edu/repository/data/bdecks_open/{year}/b{storm.lower()}{year}.dat' 
                data = pd.read_csv(link, header = None, usecols=range(20))
    return data

def getAlexStorm(storm):
    link = 'https://cyclonicwx.com/data/storms.json'
    data = urllib.urlopen(link).read().decode('utf-8')
    data = data.split('},')

    goodData = None
    for x in range(len(data)):
        if storm.upper() in data[x].upper():
            goodData = json.loads(data[x][8:] + "}")
            break 

    return goodData['lat'], 360 + goodData['lon']

# Retrieve latitude and longitude data for a given TC
def latlon(storm):
    try:
        data = getStorm(storm)
        print(data)
        lat = data[6]
        lon = data[7]

        for x in range(len(lat)):
            if ('N' in lat[x]):
                lat[x] = int(lat[x].replace('N', '')) * .1
            else:
                lat[x] = int(lat[x].replace('S', '')) * -.1
            if ('E' in lon[x]):
                lon[x] = int(lon[x].replace('E', '')) * .1
            else:
                lon[x] = 360 - int(lon[x].replace('W', '')) * .1
        return lat.iloc[-1], lon.iloc[-1]
    except:
        lat, lon = getAlexStorm(storm)
        print(lat, lon)

        return lat, lon

# Returns the RMW (radius of maximum winds) for a TC
def rmw(storm):
    data = getStorm(storm)
    return int(data[19].iloc[-1])