import glob
import time
from itertools import batched
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile


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

def ServeData(files,mean=0, rms=0,zero_correction=0):
    data = unzipper(files)
    return zero_correction - data 

def dumpOccupancyData(data,filename):
    np.save(filename+".npy",data)

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


def clustering_algo(arr,divs = 4):
    arr =pd.Series(arr,index=np.arange(0,256))
    arr_s=arr.sort_values()
    arr_s_d = np.insert(np.diff(arr_s),0,0)
    series_s_d = pd.Series(arr_s_d,index = arr_s.index)
    interval = series_s_d.sort_values()[::-1].iloc[:divs]
    bins = np.sort(arr[interval.index].values)
    bins=np.insert(bins,0,np.min(arr))
    bins=np.append(bins,np.max(arr))
    arrSeries = pd.Series(arr)
    labels=[i for i in range(0,len(np.unique(bins))-1)]
    arrSeries_cut = pd.cut(arrSeries, bins, labels=labels, duplicates='drop')

    return arrSeries_cut,labels


def strip_finding(cluster_process):
    # Calculate weighted average
    cluster_x = [item for item in cluster_process if item[0] <= 128]
    cluster_y = [item for item in cluster_process if item[0] > 128]
    
    #avgCharge_per_cluster_x = sum(value[1] for value in cluster_x) / len(cluster_x)
    #avgCharge_per_cluster_y = sum(value[1] for value in cluster_y) / len(cluster_y)
    #print(avgCharge_per_cluster_x, avgCharge_per_cluster_y)

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
    if closest_value_x == 128:
        closest_value_x = 127
    if closest_value_y ==256:
        closest_value_y = 255

    return closest_value_x, closest_value_y



def fill_occupancy_dic(files,fname=None,threshold=0.07,divs=4,zero_correction=.13):
    # totalevents = len(files)
    occupancy_arr = np.zeros((128,128))
    # occupancy_arr_file = np.zeros((128,128))
    # start_time = time.time()
    
    for iterations,file in enumerate(files):

        data = ServeData(file,zero_correction=zero_correction)
        # occupancy_arr_file = np.zeros((128,128))
        for col in data.columns:
            col_data = data[col]
            
            data_with_index = col_data.reset_index().to_numpy()
            # data_points = data_with_index[:,1].reshape(-1, 1)
            
            # cluster_labels,labels = clustering_algo(data_points.reshape(1,-1)[0],divs=divs)
            data_points = data_with_index[:,1]
            cluster_labels,labels = clustering_algo(data_points,divs=divs)
        
            cluster_labels = cluster_labels.to_numpy()
            indexes_per_cluster = {i: [] for i in range(len(labels))}
            avgCharge_per_cluster = {}
            

            i=0
            for idx, label in zip(data_points, cluster_labels):
                i+=1
                if np.isnan(label):
                    continue
                indexes_per_cluster[label].append((i,idx))


            # Print indexes for each cluster
            for cluster, indexes in indexes_per_cluster.items():
                avgCharge_per_cluster[cluster] = sum(value[1] for value in indexes_per_cluster[cluster]) / len(indexes_per_cluster[cluster])

            max_key = max(avgCharge_per_cluster, key=lambda k: avgCharge_per_cluster[k])
            cluster_process = indexes_per_cluster[max_key]

            #if avgCharge_per_cluster[max_key] < 0.96:
            if avgCharge_per_cluster[max_key] < threshold:
                continue

            stripx,stripy = strip_finding(cluster_process)
            if stripx<0 or stripy<0:
                continue
                
            # occupancy_arr_file[stripx,stripy-128] =occupancy_arr_file[stripx,stripy-128]+ 1
            occupancy_arr[stripx,stripy-128] =occupancy_arr[stripx,stripy-128]+ 1

        # occupancy_arr =np.dstack([occupancy_arr, occupancy_arr_file])

    # dumpOccupancyData(occupancy_arr,fname)

    return occupancy_arr

def analysis6mzip(function=fill_occupancy_dic, files = None, n_cores = 4):
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
    import matplotlib.pyplot as plt
    startt= time.time()
    files = input()
    cores= int(input("cores"))
    files = glob.glob(f"{files}\\*.zip")
    
    dataGrid = analysis6mzip(files = files,n_cores=cores)
    np.save(f'result', dataGrid)
    dataGrid[:,5] = 0
    dataGrid[:,65] =0
    dataGrid[65,:] = 0
    dataGrid[5,:] =0
    inds = np.where(dataGrid==0)
    result = dataGrid.copy()
    indxs = inds[0]
    indys=inds[1]
    for i,indx in enumerate(indxs):
        indy = indys[i]
        result[indx,indy] = np.sum(dataGrid[indx-1:indx+1,indy-1:indy+1])/8.0#, np.sum(dataGrid[indx+1,indy-1:indy+1]), dataGrid[indx-1,indy-1:indy+1]
    print(inds)
    endt= time.time()
    print(startt-endt)
    plt.imshow(dataGrid[::2,::2])
    plt.show()
    plt.imshow(dataGrid[1::2,1::2])
    plt.show()
