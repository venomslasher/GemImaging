import glob
import time
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


def clustering_algo(arr,divs = 4):
#     print(arr)
    arr =pd.Series(arr,index=np.arange(0,256))
    arr_s=arr.sort_values()
    arr_s_d = np.insert(np.diff(arr_s),0,0)
    series_s_d = pd.Series(arr_s_d,index = arr_s.index)
    interval = series_s_d.sort_values()[::-1].iloc[:divs]
    bins = np.sort(arr[interval.index].values)
    bins=np.insert(bins,0,np.min(arr))
    bins=np.append(bins,np.max(arr))
    # print('bins:',bins)
    #print('minmax',np.min(arr),np.max(arr))
    arrSeries = pd.Series(arr)
    labels=[i for i in range(0,len(np.unique(bins))-1)]
    arrSeries_cut = pd.cut(arrSeries, bins, labels=labels, duplicates='drop')
#     display('after cut',arrSeries_cut)
#     plt.plot(arrSeries_cut)
#     plt.plot(arr)
#     plt.show()
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


def fill_occupancy_dic(files,fname,threshold=0.07,divs=4,zero_correction=0):
    totalevents = len(files)
    printProgressBar(0, totalevents, prefix = 'Progress:', suffix = 'Complete', length = 50)
    occupancy_arr = np.zeros((128,128))
    start_time = time.time()
    
    for iterations,file in enumerate(files):

        data = ServeData(file,zero_correction=zero_correction)
        occupancy_arr_file = np.zeros((128,128))
        printProgressBar(iterations + 1, totalevents, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for col in data.columns:
            col_data = data[col]         
            
            data_with_index = col_data.reset_index().to_numpy()
            data_points = data_with_index[:,1].reshape(-1, 1)
            
            cluster_labels,labels = clustering_algo(data_points.reshape(1,-1)[0],divs=divs)
        
            cluster_labels = cluster_labels.to_numpy()
            indexes_per_cluster = {i: [] for i in range(len(labels))}
            avgCharge_per_cluster = {}

            # Iterate through data points and store indexes for each cluster
            i=0
            
            for idx, label in zip(data_points[:, 0], cluster_labels):
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
                
            occupancy_arr_file[stripx,stripy-128] =occupancy_arr_file[stripx,stripy-128]+ 1

        occupancy_arr =np.dstack([occupancy_arr, occupancy_arr_file])


    # dumpOccupancyData(occupancy_arr,fname)
    end_time = time.time()
    
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} in seconds.")
    return occupancy_arr,elapsed_time

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
