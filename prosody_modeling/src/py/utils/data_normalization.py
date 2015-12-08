import numpy as np

def store_stats(mean_arr, std_arr, dim, stats_file):
    fid2 = open(stats_file,'w');
    for x in range(dim):
        fid2.write(str(mean_arr[x])+'\n') 
    for x in range(dim):
        fid2.write(str(std_arr[x])+'\n') 
    fid2.close()

def MVN_normalize(data, dim, stats_file):
    mean_arr = [];
    std_arr  = []

    for i in range(dim):
        tg = [data[x] for x in range(len(data)) if np.mod(x,dim)==i] 
        mean_arr.append(np.mean(tg))
        std_arr.append(np.std(tg))

    store_stats(mean_arr,std_arr, dim, stats_file)    

    for i in range(len(data)):
        data[i] = (data[i] - mean_arr[np.mod(i,dim)])/std_arr[np.mod(i,dim)]

    return data

def MVN_denormalize(arr, dim, stats_file):
    ip = open(stats_file,'r')
    data_mustd = [float(x.strip()) for x in ip.readlines()]
    ip.close()
    
    for i in range(len(arr)):
        k = np.mod(i,dim)
        arr[i] = (arr[i]*data_mustd[k+dim])+data_mustd[k]
        
    return arr

def hybrid_normalize(data, dim, stats_file):
    mean_arr = [];
    std_arr  = []

    for i in range(dim):
        tg = [data[x] for x in range(len(data)) if np.mod(x,dim)==i] 
        mean_arr.append(np.mean(tg))
        std_arr.append(np.std(tg))

    store_stats(mean_arr,std_arr, dim, stats_file)    

    for i in range(len(data)):
        dim_indx = np.mod(i,dim)
        if dim_indx<10:
            data[i] = 0.99 if data[i] == 1 else 0.01
        else:
            data[i] = (data[i] - mean_arr[dim_indx])/std_arr[dim_indx]

    return data

def hybrid_denormalize(arr, dim, stats_file):
    ip = open(stats_file,'r')
    data_mustd = [float(x.strip()) for x in ip.readlines()]
    ip.close()
    
    for i in range(len(arr)):
        k = np.mod(i,dim)
        if not k<10:
            arr[i] = (arr[i]*data_mustd[k+dim])+data_mustd[k]
        
    return arr
