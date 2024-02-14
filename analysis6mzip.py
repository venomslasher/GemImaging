import glob
import time
from itertools import batched
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile
from io import TextIOWrapper, StringIO
from scipy.signal import find_peaks


MAX_CHARGE= .125


def DataCleaner(data):
    return pd.DataFrame(data).astype('float64')

def unzipper1(input_zip, delimiter=','):
    with ZipFile(input_zip, 'r') as zip_file:
        # Assuming there's only one file in the zip archive
        file_name = zip_file.namelist()[0]
        with zip_file.open(file_name) as file_input:
            # Use TextIOWrapper to handle decoding
            lines = TextIOWrapper(file_input, encoding='utf-8').readlines()

    # Filter and process lines
    data = []
    for line in lines:
        print(line[0])
        cleaned_line = line.strip().replace(',', '.')
        cleaned_line = line.strip().replace('\t', ',')
        if line and (line[0].isdigit() or (line[0] == '-' and line[1:4].isdigit())):
            data.append(cleaned_line)
        else:
            print(f"Ignoring line: {line}")

    # Create a string buffer using StringIO
    buffer = StringIO('\n'.join(data))

    # Create DataFrame directly from CSV data with configurable delimiter
    df = pd.read_csv(buffer, header=None, delimiter=delimiter)

    return df


def unzipper2(input_zip,delimiter='\t'):
    with ZipFile(input_zip, 'r') as zip_file:
        # Assuming there's only one file in the zip archive
        file_name = zip_file.namelist()[0]
        with zip_file.open(file_name) as file_input:
            # Use TextIOWrapper to handle decoding
            lines = TextIOWrapper(file_input, encoding='utf-8').readlines()

    # Filter and process lines
    data = [line.strip().replace(',', '.') for line in lines if line and (line.strip()[0].isdigit() or (line[0] == '-' and line[1:].isdigit()))]
    buffer = StringIO('\n'.join(data))

    # Create DataFrame from buffer data
    df = pd.read_csv(buffer,delimiter=delimiter, header=None)
    return df


def unzipper(input_zip):
    input_zip=ZipFile(input_zip)
    data = []
    data_df = {}
    with input_zip.open(input_zip.namelist()[0]) as file_input:
        lines = file_input.readlines()

        for line in lines:
                line=line.decode('UTF-8')
                if line.startswith('#'):
                    continue
                if line.startswith('N'):
                    continue
                if line.startswith('d'):
                    continue
                
                line = line.replace(',','.')
                line = line.replace('\t',',')
                data.append(line)
    for di,d in enumerate(data):
        data_df[di] = d.split(',')[:-1]
    df = DataCleaner(data_df).T
    return df


def std_mu(x,y= None, return_mu=False):
    if y is None:
        y = np.ones(len(x))
    mu = np.average(x,weights=y)
    x_std = np.average((x-mu)**2, weights = y)
    if return_mu:
        return mu, x_std
    return x_std


def consecutive_numbers(arr):
    consecutive_sets = np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

    # Filter sets with length greater than 1 (consecutive numbers)
    consecutive_sets = [set for set in consecutive_sets if len(set) > 1]

    return consecutive_sets


def analyser(files,cuts=.005):
    dg = np.zeros((128,128))
    for file in files:
        fname= file
        print(fname)
        data = unzipper(file)
        for nx in range(0,128):
            odx = data.iloc[nx]
            dx = odx.min()#np.sort(odx.to_numpy())[:2*HALF_WIDTH_PEAK]#.min()
            if dx<cuts:
                for ny in range(128,256):
                    ody = data.iloc[ny]
                    dy = ody.min()#np.sort(ody.to_numpy())[:2*HALF_WIDTH_PEAK]#min()
                    if dy<cuts:
                        # if any(np.abs(dx-dy) < np.abs(THRESHOLD)):
                        dg[nx, ny-128]+= 1
    return dg




def analyser2(files,cuts=.005):
    dg = np.zeros((128,128))
    for file in files:
        data_mins =MAX_CHARGE- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]   

        xp,xh = find_peaks(x_min,height=0.075)
        yp,yh = find_peaks(y_min,height=0.075)
        fname= file
        print(fname)
        data = unzipper(file)
        for nx in range(0,128):
            odx = data.iloc[nx]
            dx = odx.min()#np.sort(odx.to_numpy())[:2*HALF_WIDTH_PEAK]#.min()
            if dx<cuts:
                for ny in range(128,256):
                    ody = data.iloc[ny]
                    dy = ody.min()#np.sort(ody.to_numpy())[:2*HALF_WIDTH_PEAK]#min()
                    if dy<cuts:
                        # if any(np.abs(dx-dy) < np.abs(THRESHOLD)):
                        dg[nx, ny-128]+= 1
    return dg

def core_analyser(files):
    dg = np.zeros((128,128))
    for file in files:
        data=unzipper(file)
        
        data_mins =MAX_CHARGE- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]   

        xp,xh = find_peaks(x_min,height=0.075)
        yp,yh = find_peaks(y_min,height=0.075)
        for i,nx in enumerate(xp):       
            for j,ny in enumerate(yp):
                dg[nx, ny-128]+= (xh['peak_heights'][i]+yh['peak_heights'][j])    
        return dg



def analysis6mzip(function=analyser, files = None, n_cores = 4,cuts:float = None):
    if files is None:
        files = glob.glob("*.zip")    
    filelist = batched(files,n_cores)
    
    dataGrid = np.zeros((128,128))

    with ProcessPoolExecutor() as Executor:
        results = Executor.map(function, filelist)
    for result in results:
        dataGrid+= result
    return dataGrid



if __name__ == "__main__":

    
    HALF_WIDTH_PEAK = 4
    THRESHOLD = 0.0005
    
    cuts = float(input('cut value :'))   
    dataGrid = analysis6mzip(cuts= cuts) 
    np.save(f'thr_{cuts}', dataGrid)

    
