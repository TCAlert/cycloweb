import gzip
import pandas as pd 
from helper import strip 
from datetime import datetime 
import urllib.request as urllib
import numpy as np 

# Retrieve most the A-Deck text file from SSD for the requested storm and open it in Python
def getData(storm):
    year = str(datetime.now().year)
    if ('al' in storm.lower() or 'ep' in storm.lower() or 'cp' in storm.lower()):
        link = 'https://ospo.noaa.gov/tropical-data/ATCF/NHC/a' + storm.lower() + year + '.dat'     
        file = urllib.urlopen(link).read().decode('utf-8')    
    else:
        if ('sh' in storm.lower() and datetime.now().month >= 11):
            year = str(int(year) + 1)
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/JTWC/a' + storm.lower() + year + '.dat'    
            file = urllib.urlopen(link).read().decode('utf-8')           
        else:
            link = 'https://ospo.noaa.gov/tropical-data/ATCF/JTWC/a' + storm.lower() + year + '.dat'    
            file = urllib.urlopen(link).read().decode('utf-8')           
    print(link)

    return file

# Takes the data and converts it into a format that is easier to work with, at least for these purposes
# Could be refined further in the future
def processData(data):
    data = strip(data.split('\n'))[:-1]

    newData = []
    for x in range(len(data)):
        data[x] = strip(data[x].split(','))
        try:
            if data[x][11] == '34' or data[x][11] == '0':
                if data[x][6][-1] == 'N':
                    data[x][6] = round(float(data[x][6][:-1]) * .1, 1)
                else:
                    data[x][6] = round(float(data[x][6][:-1]) * -.1, 1)

                if data[x][7][-1] == 'E':
                    data[x][7] = round(float(data[x][7][:-1]) * .1, 1)
                else:
                    data[x][7] = round(float(data[x][7][:-1]) * -.1, 1)
                newData.append(data[x][0:11])
        except:
            pass
    return newData

# Filter data based on the requested date, models, and forecast hour
# Should be improved in the future in order to make this more modular, but this is sufficient for now
def filterData(storm, date, models, hour, df = True):
    data = processData(getData(storm))

    if df == True:
        filtered = []
        for x in range(len(data)):
            if data[x][2].strip() in date and data[x][4].strip() in models and int(data[x][5]) in hour:
                filtered.append(data[x])

        filtered = pd.DataFrame(filtered)
    else:
        filtered = [[] for model in models]
        for x in range(len(data)):
            for y in range(len(models)):
                if data[x][2].strip() in date and data[x][4].strip() == models[y] and int(data[x][5]) in hour:
                    filtered[y].append(data[x])

        for x in range(len(filtered)):
            l = len(filtered[x])
            while l < 9:
                l = l + 1
                filtered[x].append(np.full(11, np.nan)) 

        filtered = np.array(filtered, dtype = 'object')
    
    return filtered