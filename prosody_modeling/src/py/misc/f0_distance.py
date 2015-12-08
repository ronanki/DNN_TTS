import os
import sys
import numpy as np

coef_size = 9;
dim       = coef_size+5;

if __name__ == "__main__":

    #### configurations ####

    data_path   = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Feature_extraction/Output/' 
    HTK_labels  = data_path+'HTK_labels_AT/';
    inp_f0_path = data_path+'f0_s4a_AT/';
    out_f0_path = data_path+'gen/AT/IDCT_14/f0/';
    file_list   = data_path+'test_id_list.scp'
 
    ###### initializations #####
    
    diff_err = 0;
    count1=0;
    count2=0;
    count3=0;

    ##### processing list of files ######

    ip1 = open(file_list,'r')
    for k in ip1.readlines():
        fname = k.strip()
        print fname

        labfile = HTK_labels +'/'+fname+'.lab'
        in_str1 = inp_f0_path+'/'+fname+'.f0'
        in_str2 = out_f0_path+'/'+fname+'.f0'
    
        ###### read original f0 file #######

        ip1 = open(in_str1,'r')
        f0_arr1 = [float(x.strip()) for x in ip1.readlines()]
        ip1.close()

        ###### read generated f0 file #######

        ip2 = open(in_str2,'r')
        f0_arr2 = [float(x.strip()) for x in ip2.readlines()]
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
    
        count=0;
        sum = 0;
    
        ph_start = int(ph_arr[0][1]/(np.power(10,4)*5));
        ph_end   = int(ph_arr[1][file_len-2]/(np.power(10,4)*5));

        org_frame_contour = f0_arr1[ph_start:ph_end];
        gen_frame_contour = f0_arr2;

        if(len(org_frame_contour)!=len(gen_frame_contour)):
            print 'Number of lines is not matching !!'
            continue;
    
        for k in range(len(org_frame_contour)):
            if(org_frame_contour[k]>0 and gen_frame_contour[k]>0):
                count1=count1+1;
                diff_err = diff_err + np.power((org_frame_contour[k] - gen_frame_contour[k]),2);
            elif(org_frame_contour[k]==0 and gen_frame_contour[k]==0):
                count2 = count2 + 1;
        
        count3 = count3 + len(org_frame_contour);
            
    diff_err = np.sqrt(diff_err/count1)
    vuv_error = float(count3 - count2 - count1)/float(count3)*100

    print 'diff_err '+str(diff_err)
    print 'vuv_error '+str(vuv_error)
    print count3-count2-count1

    #minf0_all
    #maxf0_all
    #mean_frames = sum/count;
