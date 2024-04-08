import glob
from itertools import batched
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile
from io import TextIOWrapper, StringIO
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


MAX_CHARGE= .125


def discretise(arr, h=0):
    temp_arr = arr.copy()
    
    arr_min = 0
    arr_max = 0.125
    r = arr_max - arr_min
    w = r/h
    ih=0
    while ih<=h:
        temp_arr[np.where((arr_min+(ih*w)<arr) & (arr <arr_min+(ih+1)*w))] = ih
        ih+=1
    temp_arr[temp_arr>.125] = ih+1

    return temp_arr


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


#0.075
def core_analyser(files,height_threshold=0.07,max_charge=MAX_CHARGE,peakwidth = 8):
    dg = np.zeros((128,128))
    for file in files:
        data=unzipper(file)
        
        data_mins =max_charge- data.min(axis=1)
        x_min = data_mins.iloc[:128]
        y_min = data_mins.iloc[128:]   

        xp,xh = find_peaks(x_min,height=height_threshold,width=peakwidth)
        yp,yh = find_peaks(y_min,height=height_threshold,width=peakwidth)

        xv = xh['peak_heights']#discretise(xh['peak_heights'])
        yv = yh['peak_heights']#discretise(yh['peak_heights'])

        bins = np.arange(0,.125,0.05)
        bin_label = bins[1:]

       
        for i,nx in enumerate(xp):       
            for j,ny in enumerate(yp):
                # if np.abs(xv[i]/yv[j])>.75:
                dg[nx, ny-128]+= 1#xv[i]+yv[j]
        return dg



def strip_finding(cluster_process):
    # Calculate weighted average
    cluster_x = [item for item in cluster_process if item[0] <= 128] 
    cluster_y = [item for item in cluster_process if item[0] > 128] 
    
    weighted_sum_x, weighted_sum_y = sum(value * weight for value, weight in cluster_x), sum(value * weight for value, weight in cluster_y)
    total_weight_x, total_weight_y = sum(weight for _, weight in cluster_x), sum(weight for _, weight in cluster_y)
    
    if (total_weight_x!=0 and total_weight_y!=0):
        weighted_avg_x, weighted_avg_y = weighted_sum_x / total_weight_x, weighted_sum_y / total_weight_y
        # Find the first value closest to the weighted average
        closest_value_x = min(cluster_x, key=lambda x: abs(x[0] - weighted_avg_x))[0]
        closest_value_y = min(cluster_y, key=lambda y: abs(y[0] - weighted_avg_y))[0]
    else:
        closest_value_x=-1
        closest_value_y=-1
    
    return closest_value_x, closest_value_y


def image_reconstructor(files):
    dg = np.zeros((128,128))
    for file in files:
        data=unzipper(file)
        inverted_data = 1 - data
        transpose_inverted_data = inverted_data.T
        for index, d_ in transpose_inverted_data.iterrows():
            data_ = d_.values
            data_ = data_.reshape(-1,1)
            kmeans = KMeans(n_clusters=4, random_state=42,n_init=10)
            kmeans.fit(data_)
            centers = kmeans.cluster_centers_
            cluster_labels = kmeans.labels_

            charges_per_cluster = {i: [] for i in range(kmeans.n_clusters)}
            avgCharge_per_cluster = {}
            for i, charge in enumerate(data_):
                charges_per_cluster[cluster_labels[i]].append((i,charge))

            for cluster_label, charges in charges_per_cluster.items():
                avgCharge_per_cluster[cluster_label] = sum(value[1] for value in charges_per_cluster[cluster_label]) / len(charges_per_cluster[cluster_label])
            
            max_avgCharge_key = max(avgCharge_per_cluster, key=lambda k: avgCharge_per_cluster[k])
            cluster_to_process = charges_per_cluster[max_avgCharge_key]

            if avgCharge_per_cluster[max_avgCharge_key] < 0.91:
                continue

            stripx, stripy = strip_finding(cluster_to_process)
            if stripx==-1 or stripy==-1:
                continue

            dg[stripx,stripy-128] +=1

    return dg




def analysis6mzip(function=core_analyser, files = None, n_cores = 4):
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
    dataGrid = analysis6mzip() 
    np.save(f'result', dataGrid)

    
