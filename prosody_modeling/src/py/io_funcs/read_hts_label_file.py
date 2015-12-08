import os
import numpy as np
from binary_io import BinaryIOCollection


class readHTSlabelFile(object):

    def read_state_align_label_file(self, lab_file):
        
        ###### read HTS state-align lab file ########
        ip = open(lab_file, 'r')
        data = ip.readlines()
        sz = len(data) / 5
        ip.close()
    
        ### Initialisation of variables ###
        mean_f0_arr = self.zeros(sz, 8)
        phone = [];ph_arr = [];st_arr = []
        lc = 1;np_wrd = 0;np_phr = 0
        ph_arr.append([])
        ph_arr.append([])
        
        ip = open(lab_file, 'r')
        for x in ip.readlines():
            fstr = x.strip().split()
            if(np.mod(lc, 5) == 1):
                ph_start = fstr[0]
                st_arr.append([])
            if(np.mod(lc, 5) == 0):
                ph_end = fstr[1]
                ftag = fstr[2]
                ph = ftag[ftag.index('-') + 1:ftag.index('+')]
                phone.append(ph)
                ph_arr[0].append(float(ph_start))
                ph_arr[1].append(float(ph_end))
    
                if(ph == 'pau'):
                    for j in range(8):
                        mean_f0_arr[(lc - 1) / 5][j] = (lc - 1) / 5
    
                if(ph != '#' and ph != 'pau'): 
    
                    ### phone information ###
                    mean_f0_arr[(lc - 1) / 5][0] = (lc - 1) / 5
                    mean_f0_arr[(lc - 1) / 5][1] = (lc - 1) / 5
    
                    ### syllable information ###
                    ss = ftag[ftag.index(':') + 1:ftag.index('_')]
                    se = ftag[ftag.index('_') + 1:ftag.index('/')]
                    if(int(se) == 1):
                        mean_f0_arr[(lc - 1) / 5][2] = ((lc - 1) / 5) - (int(ss) - 1)
                        mean_f0_arr[(lc - 1) / 5][3] = (lc - 1) / 5
                        np_wrd = np_wrd + int(ss)
                        np_phr = np_phr + int(ss)
    
                    ### word information ###
                    we = ftag[ftag.index('&') - 1:ftag.index('&')]
                    if(int(se) == 1 and int(we) == 1):
                        mean_f0_arr[(lc - 1) / 5][4] = ((lc - 1) / 5) - (np_wrd - 1)
                        mean_f0_arr[(lc - 1) / 5][5] = (lc - 1) / 5
                        np_wrd = 0
    
                    ### phrase information ###
                    ft_in = ftag.index('&')
                    ft_fn = ftag.index('/C/')
                    cur_phr = ftag[ft_in + 1:ft_fn]
    
                    pe = cur_phr[cur_phr.index('-') + 1:cur_phr.index('#')]
                    if(int(se) == 1 and int(we) == 1 and int(pe) == 1):
                        mean_f0_arr[(lc - 1) / 5][6] = ((lc - 1) / 5) - (np_phr - 1)
                        mean_f0_arr[(lc - 1) / 5][7] = (lc - 1) / 5
                        np_phr = 0
    
            st_arr[(lc - 1) / 5].append(int(fstr[1]) - int(fstr[0]))
            lc = lc + 1
    
        ip.close()
        file_len = len(phone)
    
        i = file_len - 2
        while(i > 0):
            for j in range(4):
                if(int(mean_f0_arr[i][2 * j]) == 0):
                    mean_f0_arr[i][2 * j] = mean_f0_arr[i + 1][2 * j]
                    mean_f0_arr[i][2 * j + 1] = mean_f0_arr[i + 1][2 * j + 1]
            i = i - 1
            
        return phone, ph_arr, mean_f0_arr
    
    def zeros(self, m, n):
        if(n == 1):
            arr = np.ndarray((m,), float)
        else:
            arr = np.ndarray((m, n), float)
        arr.fill(0)
        return arr
    
if __name__ == "__main__":
    dnn_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/dnn_tts_blzpilot/'
    label_align_dir = os.path.join(dnn_dir, 'two_stage_mtldnn/data/label_state_align')  
    
    htsclass = readHTSlabelFile()
    io_funcs = BinaryIOCollection()
    
    DFP = 1
    if DFP:
        parseLabFile = True;
        
        if parseLabFile:
            filelist = os.path.join(dnn_dir, 'two_stage_mtldnn/data/file_id_list.scp')
            list_of_files = io_funcs.load_file_list(filelist)
            
            max_syl_dur = 0
            max_syl_dur_filename = ''
            for i in range(len(list_of_files)):
                filename = list_of_files[i]
                print filename
                
                in_lab_file = os.path.join(label_align_dir, filename + '.lab')
                [phone, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
                
                for j in range(len(phone)):
                    num_of_phones = (mean_f0_arr[j][5] - mean_f0_arr[j][4]) + 1
                    if(num_of_phones>max_syl_dur):
                        max_syl_dur=num_of_phones
                        max_syl_dur_filename = in_lab_file
            print max_syl_dur
            print max_syl_dur_filename