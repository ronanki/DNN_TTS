import os
import sys
import scipy as sp
import numpy as np
import scipy.fftpack as fftpack
import scipy.interpolate

coef_size = 9;
dim       = coef_size+5;

def dct(arr):
    dcta2 = fftpack.dct(arr,norm='ortho')
    return dcta2

def idct(arr):
    idcta2 = fftpack.idct(arr,norm='ortho')
    return idcta2

def MVN_denormalize(arr,stats_file):
    ip = open(stats_file,'r')
    data_mustd = [float(x.strip()) for x in ip.readlines()]
    ip.close()
    for i in range(len(arr)):
        k = np.mod(i,dim)
        arr[i] = (arr[i]*data_mustd[k+dim])+data_mustd[k]
    return arr    
        
def interpolation_with_zeros(arr,n):
    xlen = len(arr)
    xtralen = n - xlen
    for x in range(xtralen):
        arr.append(0)
    return arr

def idct_with_size(arr,n):
    xlen = len(arr)
    if n>xlen:
        new_arr = interpolation_with_zeros(arr,n)
        idcta2 = idct(new_arr)
    else:
        new_arr = arr[0:n]
        idcta2 = idct(new_arr)
    return idcta2

def zeros(m,n):
    if(n==1):
        arr = np.ndarray((m,),float)
    else:
        arr = np.ndarray((m,n),float)
    arr.fill(0)
    return arr

if __name__ == "__main__":

    #### configurations ####

    data_path   = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Feature_extraction/Output/' 
    f0_contour  = data_path+'gen/AT/f0_contour_dct_s4a/';
    HTK_labels  = data_path+'HTK_labels_AT/';
    inp_f0_path = data_path+'f0_s4a_AT/';
    out_f0_path = data_path+'gen/AT/IDCT_14/f0/';
    dur_path    = data_path+'gen/AT/IDCT_14/dur/';
    file_list   = data_path+'test_id_list.scp'
    stats_file  = data_path+'mean_stats'
    
    normalization = "MVN"

    ##### processing list of files ######

    ip1 = open(file_list,'r')
    for k in ip1.readlines():
        fname = k.strip()
        print fname

        labfile  = HTK_labels +'/'+fname+'.lab'
        f0file   = inp_f0_path+'/'+fname+'.f0'
        in_str1  = f0_contour +'/'+fname+'.cmp'

        out_str1 = out_f0_path+'/'+fname+'.f0'
        out_str2 = dur_path   +'/'+fname+'.dur'
    
        fid_out1 = open(out_str1,'w');
        fid_out2 = open(out_str2,'w');

        ###### read generated cmp file #######

        ip1 = open(in_str1,'r')
        gen_data = [float(x.strip()) for x in ip1.readlines()]
        ip1.close()

        no_of_valid_phones = len(gen_data)/(dim);

        ##### denormalize the data #####

        print 'De-Normalizing the data....'
        if(normalization=="MVN"):
            denorm_data = MVN_denormalize(gen_data,stats_file);
        else:
            denorm_data = gen_data

        ###### read input f0 file #######

        ip2 = open(f0file,'r')
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
    
        ph_cnt=0;
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
            frame_contour = [e for j, e in enumerate(org_frame_contour) if e != 0]
            no_of_frames = len(frame_contour);
            gen_frame_contour = zeros(org_no_of_frames,1);

            st_pt = dim*(ph_cnt);
            fn_pt = dim*(ph_cnt+1);

            gen_frame_data = denorm_data[st_pt:fn_pt];

            if no_of_frames>3:  
                est_mean = (gen_frame_data[10])*float(no_of_frames)*2*np.sqrt(1/(4*float(no_of_frames)));
                gen_frame_data[0] = est_mean
                gen_norm_f0 = idct_with_size(gen_frame_data[0:coef_size],no_of_frames);

                gen_f0 = np.power(10,gen_norm_f0)
                gen_f0 = (np.array(gen_f0)-1)/0.00437;
                for x in range(len(gen_f0)):
                    if gen_f0[x]<50:
                        gen_f0[x] = 0;
                count=0;        
                for x in range(len(org_frame_contour)):         
                    if(org_frame_contour[x]>0):
                        gen_frame_contour[x] = gen_f0[count];
                        count=count+1

            ph_cnt = ph_cnt + 1
            for j in range(org_no_of_frames):
                    fid_out1.write(str(gen_frame_contour[j])+'\n');
            fid_out2.write(str((np.power(10,gen_frame_data[dim-1])-1)/0.00437)+'\n');
    
        fid_out1.close()
        fid_out2.close()
        ### uncomment below line while you are running for first time
        #break; ### breaks after processing one file - to check errors
