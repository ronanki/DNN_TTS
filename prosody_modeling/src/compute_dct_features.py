import os
import sys
import scipy as sp
import numpy as np
import scipy.fftpack as fftpack
import scipy.interpolate

coef_size = 9;
dim       = coef_size + 5

def dct(arr):
    dcta2 = fftpack.dct(arr,norm='ortho')
    return dcta2

def mean_interpolation(x,y):
    y_interp = scipy.interpolate.interp1d(x, y)
    return y_interp

def mean_fit(arr):
    xlen = len(arr)
    x = [i for i in range(xlen)]
    y = arr
    y_interp = mean_interpolation(x,y)
    new_arr = [y[i] for i in range(xlen-1)]
    xtralen= coef_size-xlen
    
    for i in range(xtralen):
        value = xlen-2+1/(float(xtralen)+1)*(i+1)
        new_arr.append(y_interp(value))
    new_arr.append(y[xlen-1])    
    return new_arr  

def interpolation_with_zeros(arr):
    xlen = len(arr)
    xtralen = coef_size - xlen
    for x in range(xtralen):
        arr.append(0)
    return arr

def store_stats(mean_arr,std_arr,stats_file):
    fid2 = open(stats_file,'w');
    for x in range(dim):
        fid2.write(str(mean_arr[x])+'\n') 
    for x in range(dim):
        fid2.write(str(std_arr[x])+'\n') 
    fid2.close()

def MVN_normalize(data,stats_file):
    mean_arr = [];
    std_arr  = []

    for i in range(dim):
        tg = [data[x] for x in range(len(data)) if np.mod(x,dim)==i] 
        mean_arr.append(np.mean(tg))
        std_arr.append(np.std(tg))

    store_stats(mean_arr,std_arr,stats_file)    

    for i in range(len(data)):
        data[i] = (data[i] - mean_arr[np.mod(i,dim)])/std_arr[np.mod(i,dim)]
    
    return data
    

if __name__ == "__main__":

    #### configurations ####

    data_path = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Feature_extraction/Output/'

    HTK_labels = data_path+'HTK_labels_AT/';
    f0_path    = data_path+'f0_AT/';
    f0_contour = data_path+'f0_contour_dct_AT/';
    file_list  = data_path+'file_id_list.scp'
    stats_file = data_path+'mean_stats'

    normalization = "MVN"

    ###### initializations #####

    ph_count   = 0;
    data_mustd = [];
    flens      = []

    ##### processing list of files ######

    ip1 = open(file_list,'r')
    for k in ip1.readlines():
        fname = k.strip()
        print fname
    
        labfile = HTK_labels+'/'+fname+'.lab'
        in_str  = f0_path   +'/'+fname+'.lf0'
    
        ###### read input f0 file #######

        ip2 = open(in_str,'r')
        f0_arr = [float(x.strip()) for x in ip2.readlines()]
        ip2.close()

        ###### read lab file ########

        phone = []
        ph_arr = []
        ph_arr.append([])
        ph_arr.append([])
        ip3 = open(labfile,'r');
        for x in ip3.readlines():
            fstr = x.strip().split()
            phone.append(fstr[2])
            ph_arr[0].append(float(fstr[0]))
            ph_arr[1].append(float(fstr[1]))

        ip3.close();
        
        file_len = len(phone);
    
        ##### process each phoneme in lab file #####

        sil_count=0;
        for i in range(file_len): 
            frame_contour = [];frame_dct = [];
        
            #### do nothing for silence ####

            if(phone[i] == '#'):
                sil_count = sil_count+1
                continue;

            ph_start = int(ph_arr[0][i]/(np.power(10,4)*5));
            ph_end   = int(ph_arr[1][i]/(np.power(10,4)*5));

            org_no_of_frames = ph_end - ph_start;
            org_frame_contour = f0_arr[ph_start:ph_end];
            frame_contour = [e for i, e in enumerate(org_frame_contour) if e != 0]
            no_of_frames = len(frame_contour);

            min_f0=0;max_f0=0;mean_f0=0;var_f0=0;
            frame_mxfit = np.ndarray((10,),int)
            frame_mxfit.fill(0)
        
            if(no_of_frames>3):
                min_f0 = np.min(frame_contour);
                max_f0 = np.max(frame_contour);
                mean_f0 = np.mean(frame_contour);
                var_f0 = np.std(frame_contour);
                
                frame_contour_norm = np.log10(0.00437*np.array(frame_contour)+1);
                frame_dct = dct(frame_contour_norm);

                ###### average fitting algorithm ######

                if(no_of_frames<coef_size):
                    frame_mxfit = mean_fit(frame_dct)
                elif(no_of_frames>coef_size):
                    frame_mxfit = frame_dct[0:coef_size]
                else:
                    frame_mxfit = frame_dct

                if(len(frame_mxfit)!=coef_size):
                    print 'Error in average fitting algorithm'
                    sys.exit(1)
                
            ######### storing feature values #############

            for j in range(coef_size):
                data_mustd.append(frame_mxfit[j])

            data_mustd.append(np.log10(min_f0*0.00437+1))
            data_mustd.append(np.log10(mean_f0*0.00437+1))
            data_mustd.append(np.log10(max_f0*0.00437+1))
            data_mustd.append(var_f0/10)
            data_mustd.append(np.log10(org_no_of_frames*5*0.00437+1))
              
            ph_count=ph_count+1
        
        if(sil_count>2):
            print 'silence count in this file is greater than 2: '+fname

        flens.append((file_len-sil_count)*14)
        
        #break;

    ip1.close()

    ##### normalize the data #####

    if(normalization=="MVN"):
        norm_data = MVN_normalize(data_mustd,stats_file)
    else:
        norm_data = data_mustd

    ##### write features into files #####

    ip1 = open(file_list,'r')
    count=0;idx=0;flength=0;
    for k in ip1.readlines():
        fname = k.strip()
        print fname
        out_str = f0_contour+'/'+fname+'.cmp'
        op1 = open(out_str,'w');
        flength = flength+flens[count]
        while idx < flength:
            op1.write(str(norm_data[idx])+'\n')
            idx=idx+1
        op1.close();
        count=count+1
        #break;

    ip1.close()

        
