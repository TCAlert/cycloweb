import os
import urllib.request
import gzip

OUTPUTS = os.environ.get('CYCLOBOT_OUTPUTS', r'C:\Users\deela\Downloads')

def getFTP(link):
    name = link.split('/')
    # Local path where you want to save the downloaded file
    local_filename = os.path.join(OUTPUTS, name[-1])

    # Download the file from the FTP server
    urllib.request.urlretrieve(link, local_filename)

    print(f"File '{local_filename}' downloaded successfully.")

def getGZ(file, newFile = None):
    if newFile is None:
        newFile = file[:-3]

    with gzip.open(file, 'rb') as f:
        with open(newFile, 'wb') as nf:
            print(f'File {file} opened successfully.')
            nf.write(f.read())

def getGRIB(link, title = None):
    if title == None:
        name = link.split('/')
        # Local path where you want to save the downloaded file
        local_filename = os.path.join(OUTPUTS, name[-1])
    else:
        local_filename = os.path.join(OUTPUTS, title)
        
    # Download the file from the FTP server
    urllib.request.urlretrieve(link, local_filename)

    print(f"File '{local_filename}' downloaded successfully.")

    return local_filename

#getFTP('ftp://ftp.ifremer.fr/ifremer/argo/dac/coriolis/6902919/profiles/R6902919_183.nc')

