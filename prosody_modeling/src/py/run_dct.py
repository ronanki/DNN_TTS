'''
Created on 9 Oct 2015

@author: Srikanth Ronanki
'''
import os
import numpy as np
from models.dct_models import DCTModels
from io_funcs.binary_io import BinaryIOCollection
# from utils import f0_processing
from utils import eval_metrics
from utils import data_normalization

import matplotlib.pyplot as plt
# import matplotlib
# import math
# import matplotlib.ticker as ticker

class DCTFeatures:
    def __init__(self):
        self.dct_models = DCTModels()
        pass
    
    def DCT_decomposition(self, phone, ph_arr, mean_f0_arr, f0_arr, decomposition_unit, coef_size):
        ### initialisations ###
        dct_features = []; stat_features = [];
        file_len = len(phone)
        i = 0;
        while i < file_len:
            segment_contour = []; 
            #### do nothing for silence ####
            #if(phone[i] == '#'):
            #    i=i+1
            #    #sil_count = sil_count+1
            #    continue;
            if(decomposition_unit == "phone"):
                st_ph_indx = int(mean_f0_arr[i][0]);
                fn_ph_indx = int(mean_f0_arr[i][1]);
            
            if(decomposition_unit == "syllable"):
                st_ph_indx = int(mean_f0_arr[i][2]);
                fn_ph_indx = int(mean_f0_arr[i][3]);
                
            if(decomposition_unit == "word"):
                st_ph_indx = int(mean_f0_arr[i][4]);
                fn_ph_indx = int(mean_f0_arr[i][5]);
                
            if(st_ph_indx == 0 or fn_ph_indx == 0):
                st_ph_indx = i
                fn_ph_indx = i

            ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
            ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
            
            
            num_of_frames = ph_end - ph_start;
            segment_contour = f0_arr[ph_start:ph_end];    
            # voiced_contour = [e for x, e in enumerate(segment_contour) if e != 0]
            segment_contour_norm = np.log10(0.00437 * np.array(segment_contour) + 1);
            dct_coeff = self.dct_models.dct(segment_contour_norm, coef_size);
                
            min_f0 = np.min(segment_contour_norm);    
            max_f0 = np.max(segment_contour_norm);
            mean_f0 = np.mean(segment_contour_norm);
            var_f0 = np.std(segment_contour_norm);
            
            for j in range(coef_size):
                dct_features.append(dct_coeff[j])
            stat_features.append(min_f0)
            stat_features.append(mean_f0)
            stat_features.append(max_f0)
            stat_features.append(var_f0)
            stat_features.append(num_of_frames)
            if(decomposition_unit == "phone"):
                i = i + 1
            elif(decomposition_unit == "syllable" or decomposition_unit == "word"):
                i = i + 1 + (fn_ph_indx - st_ph_indx)
                
        return dct_features, stat_features
    
    def DCT_reconstruction(self, dct_features, stat_features, coef_size):
        recons_f0_contour = []
        num_of_phones = len(dct_features) / coef_size
        stat_size = len(stat_features) / num_of_phones
        st_pnt = 0;fn_pnt = 0
        for i in range(num_of_phones):
            st_pnt = i * coef_size
            fn_pnt = (i + 1) * coef_size
            dct_contour = dct_features[st_pnt:fn_pnt]
            num_of_frames = stat_features[((i + 1) * stat_size) - 1]
            est_mean = (1.0 * stat_features[(i * stat_size) + 1]) * float(num_of_frames) * 2 * np.sqrt(1 / (4 * float(num_of_frames)));
            dct_contour[0] = est_mean
            gen_norm_f0 = self.dct_models.idct(dct_contour, num_of_frames)
            gen_f0 = np.power(10, gen_norm_f0)
            gen_f0 = (np.array(gen_f0) - 1) / 0.00437;
            for j in range(len(gen_f0)):
                recons_f0_contour.append(gen_f0[j])
        return recons_f0_contour
    
    def DCT_decomposition_from_lab_file(self, phone, ph_arr, mean_f0_arr, f0_arr, decomposition_unit, coef_size, stat_size):
        ### initialisations ###
        dct_features = []; stat_features = [];
        file_len = len(phone)
        sil_count = 0; i = 0;
        while i < file_len:
            segment_contour = []; 
            #### do nothing for silence ####
            if(phone[i] == '#'):
                sil_count = sil_count + 1
                i = i + 1
                continue;
            if(decomposition_unit == "phone"):
                st_ph_indx = int(mean_f0_arr[i][0]);
                fn_ph_indx = int(mean_f0_arr[i][1]);
                
            if(decomposition_unit == "syllable"):
                st_ph_indx = int(mean_f0_arr[i][2]);
                fn_ph_indx = int(mean_f0_arr[i][3]);
                
            if(decomposition_unit == "word"):
                st_ph_indx = int(mean_f0_arr[i][4]);
                fn_ph_indx = int(mean_f0_arr[i][5]);
                if(phone[i] == 'pau'):
                    i = i + 1
                    continue;
            ### for silence and pause regions ###    
            if(st_ph_indx == 0 or fn_ph_indx == 0):
                st_ph_indx = i
                fn_ph_indx = i

            ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
            ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
            
            
            num_of_frames = ph_end - ph_start;
            segment_contour = f0_arr[ph_start:ph_end];    
            # voiced_contour = [e for x, e in enumerate(segment_contour) if e != 0]
            segment_contour_norm = np.log10(0.00437 * np.array(segment_contour) + 1);
            dct_coeff = self.dct_models.dct(segment_contour_norm, coef_size);
                
            min_f0 = np.min(segment_contour_norm);    
            max_f0 = np.max(segment_contour_norm);
            mean_f0 = np.mean(segment_contour_norm);
            var_f0 = np.std(segment_contour_norm);
            
            if stat_size == 10:
                delta_f0 = max_f0 - min_f0;
                dr_maxf0 = segment_contour.index(np.max(segment_contour)) + 1
                df_maxf0 = num_of_frames - dr_maxf0;
                ar_maxf0 = max_f0 - segment_contour_norm[0]
                af_maxf0 = max_f0 - segment_contour_norm[-1]
                amp_tilt = (ar_maxf0 - af_maxf0) / (ar_maxf0 + af_maxf0)
                if(ar_maxf0 + af_maxf0 == 0):
                    amp_tilt = 0
                dur_tilt = (dr_maxf0 - df_maxf0) / (dr_maxf0 + df_maxf0)
            
            for j in range(coef_size):
                dct_features.append(dct_coeff[j])
            
            ### statistical features ###
            
            stat_features.append(min_f0)
            stat_features.append(mean_f0)
            stat_features.append(max_f0)
            stat_features.append(var_f0)
            
            ### tilt features ###
            if stat_size == 10:
                stat_features.append(delta_f0)
                stat_features.append(dr_maxf0)
                stat_features.append(ar_maxf0)
                stat_features.append(amp_tilt)
                stat_features.append(dur_tilt)
            
            stat_features.append(num_of_frames)
            
            if(decomposition_unit == "phone"):
                i = i + 1
            elif(decomposition_unit == "syllable" or decomposition_unit == "word"):
                i = i + 1 + (fn_ph_indx - st_ph_indx)
                
        return dct_features, stat_features

    def DCT_reconstruction_from_lab_file(self, phone, ph_arr, mean_f0_arr, interp_f0_arr, denorm_data, decomposition_unit, coef_size, out_dim, use_org_dur):
        recons_f0_contour = []
        file_len = len(phone)
        sil_count = 0; i = 0;
        seg_count = 0
        while i < file_len: 
            if(decomposition_unit == "phone"):
                st_ph_indx = int(mean_f0_arr[i][0]);
                fn_ph_indx = int(mean_f0_arr[i][1]);
            
            if(decomposition_unit == "syllable"):
                st_ph_indx = int(mean_f0_arr[i][2]);
                fn_ph_indx = int(mean_f0_arr[i][3]);
                    
            if(decomposition_unit == "word"):
                st_ph_indx = int(mean_f0_arr[i][4]);
                fn_ph_indx = int(mean_f0_arr[i][5]);
                
            ### for silence and pause regions ###    
            if(st_ph_indx == 0 or fn_ph_indx == 0):
                st_ph_indx = i
                fn_ph_indx = i

            ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
            ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
            num_of_frames = ph_end - ph_start;
            
            #### take zeros for silence ####
            if(phone[i] == '#'):
                sil_count = sil_count + 1
                i = i + 1
                continue;
            if (decomposition_unit == 'word' and phone[i] == 'pau'):
                sil_contour = self.zeros(num_of_frames, 1)
                recons_f0_contour = np.concatenate((recons_f0_contour, sil_contour), axis=0)
                i = i + 1
                continue;
            
            ### to check rmse using original mean f0 and/or contour shape ### 
            segment_contour = interp_f0_arr[ph_start:ph_end];    
            segment_contour_norm = np.log10(0.00437 * np.array(segment_contour) + 1);
            org_dct_coeff = self.dct_models.dct(segment_contour_norm, coef_size);
            org_mean_f0 = np.mean(segment_contour_norm);
            
            st_pt = out_dim * (seg_count);
            fn_pt = out_dim * (seg_count + 1);

            gen_frame_data = denorm_data[st_pt:fn_pt];

            pred_dur = np.int(gen_frame_data[-1] + 0.5)
            if(pred_dur <= 4):
                pred_dur = 5

            #est_mean = org_mean_f0
            est_mean = gen_frame_data[coef_size + 1]
            est_mean = (1.0 * est_mean) * float(num_of_frames) * 2 * np.sqrt(1 / (4 * float(num_of_frames)));
            gen_frame_data[0] = est_mean
            #gen_frame_data[1:coef_size] = org_dct_coeff[1:coef_size]
            gen_norm_f0 = self.dct_models.idct(gen_frame_data[0:coef_size], num_of_frames);

            gen_f0 = np.power(10, gen_norm_f0)
            gen_f0 = (np.array(gen_f0) - 1) / 0.00437;
            for x in range(len(gen_f0)):
                if gen_f0[x] < 0:
                    gen_f0[x] = 0;
            
            recons_f0_contour = np.concatenate((recons_f0_contour, gen_f0), axis=0)
            
            if(decomposition_unit == "phone"):
                i = i + 1
            elif(decomposition_unit == "syllable" or decomposition_unit == "word"):
                i = i + 1 + (fn_ph_indx - st_ph_indx)
            seg_count += 1
                
        return recons_f0_contour 
    
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
    
    def interpolate_f0(self, f0_file):
        
        io_funcs = BinaryIOCollection()
        data = io_funcs.load_float_file(f0_file)
        ip_data = data

        frame_number = len(data)
        last_value = 0.0
        for i in xrange(frame_number):
            if data[i] <= 0.0:
                j = i + 1
                for j in range(i + 1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number - 1:
                    if last_value > 0.0:
                        step = (data[j] - data[i - 1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i - 1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return  ip_data
    
    def plot_dct(self, dct_features):
        ### plot ###
        dct_mean_arr = []; dct_coef_arr = []
        for i in range(len(dct_features)):
            if(i % coef_size == 0):
                dct_mean_arr.append(dct_features[i])
            else:
                if(i % coef_size != 1):
                    dct_mean_arr.append(0)
                dct_coef_arr.append(dct_features[i])
        
        ax = plt.figure(1).add_subplot(2, 1, 1)
        ax.plot(dct_mean_arr, label='DCT Mean (c0)')
        ax.set_xlim([1, len(dct_coef_arr)])
        plt.legend(loc="upper right")
        
        ax = plt.figure(1).add_subplot(2, 1, 2)
        ax.plot(dct_coef_arr, label='DCT features (6)')
        ax.set_xlim([1, len(dct_coef_arr)])
        plt.legend(loc="upper right")
        
        plt.title('DCT decomposition')
        plt.xlabel("DCT coefficients (phone level)")
        # plt.ylabel("F0 (Hz)")
        plt.legend(loc="upper right")
        plt.show()
        
    def plot_DBR(self, interp_f0_arr, recons_f0_contour):
        ### plot ###
        ax = plt.figure(1).add_subplot(1, 1, 1)
        ax.plot(interp_f0_arr, label='interpolated f0')
        ax.set_xlim([1, len(interp_f0_arr)])
        plt.legend(loc="upper right")
        
        ax = plt.figure(1).add_subplot(1, 1, 1)
        ax.plot(recons_f0_contour, label='reconstructed f0')
        ax.set_xlim([1, len(recons_f0_contour)])
        plt.legend(loc="upper right")
        
        ax.set_ylim([0, 300])
        plt.title('DCT decomposition by reconstruction')
        plt.xlabel("time (in frames)")
        plt.ylabel("F0 (Hz)")
        plt.legend(loc="upper right")
        plt.show()        
    
            
if __name__ == "__main__":
    ###    This is main function   ###    
    ### load all modules ###
    prosody_funcs = DCTFeatures()
    io_funcs = BinaryIOCollection()
    
    ### model parameters ###
    normalization = 'MVN'
    decomposition_unit = 'syllable' 
    coef_size = 9
    stat_size = 10
    out_dim = coef_size + stat_size
    in_dim = 592
    
    ### Relative work path ###
    # work_dir = os.path.join(os.getcwd(), "../../")
    
    ### Absolute work path ###
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model/'
    feat_dir_path = 'dct_' + decomposition_unit + '_' + str(coef_size) + '_stat_' + str(stat_size)
    f0_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/f0/')
    lab_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/label_state_align/')
    out_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/dct_features/' + feat_dir_path + '/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ### Single file processing ###
    SFP = 0
    if SFP:
        # filename = 'TheHareandtheTortoise_097_24-23-1-2-1-0'
        filename = 'GoldilocksandtheThreeBears_000_24-0-24-3-0-2'
            
        f0_file = os.path.join(f0_dir, filename + '.f0')
        lab_file = os.path.join(lab_dir, filename + '.lab') 
        out_file = os.path.join(out_dir, filename + '.cmp')
        
        ### processing f0 file ###
        ori_f0_data = io_funcs.load_float_file(f0_file)  # # to process float f0 file
        # ori_lf0_data, frame_number = io_funcs.load_binary_file_frame(lf0_file, 1)
        # ori_lf0_data = np.exp(ori_f0_data)
        # interp_f0_arr = f0_processing.process(ori_f0_data)
        interp_f0_arr = prosody_funcs.interpolate_f0(f0_file)
        
        ### read label file ###
        [phone, ph_arr, mean_f0_arr] = prosody_funcs.read_state_align_label_file(lab_file)
        
        ### decomposition of f0 file ###
        [dct_features, stat_features] = prosody_funcs.DCT_decomposition(phone, ph_arr, mean_f0_arr, interp_f0_arr, decomposition_unit, coef_size)
        prosody_feats = np.concatenate((dct_features, stat_features), axis=0)
        # io_funcs.array_to_binary_file(dct_features, out_file)
        
        ### reconstruction of f0 file ###
        recons_f0_contour = prosody_funcs.DCT_reconstruction(dct_features, stat_features, coef_size)
        
        ### evaluation metrics ###
        print 'RMSE: ' + str(eval_metrics.rmse(recons_f0_contour, interp_f0_arr[0:len(recons_f0_contour)]))
        print 'CORR: ' + str(eval_metrics.corr(recons_f0_contour, interp_f0_arr[0:len(recons_f0_contour)]))
        
        ### plot ###
        prosody_funcs.plot_dct(dct_features)       
        prosody_funcs.plot_DBR(interp_f0_arr, recons_f0_contour)
    
    ### Directory of files processing ###
    DFP = 1
    if DFP:
        prosodydecomp = False;
        prosodyrecons = True;
        computermse = True;
        
        stat_fname = feat_dir_path + '.txt'
        if prosodydecomp:
            stats_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', stat_fname)
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            prosody_feats = []; flens = [];
            for k in range(len(list_arr)):
                filename = list_arr[k]
                print filename
                f0_file = os.path.join(f0_dir, filename + '.f0')
                lab_file = os.path.join(lab_dir, filename + '.lab') 
                
                ### processing lf0 file ###
                # ori_f0_data, frame_number = io_funcs.load_binary_file_frame(f0_file, 1)
                # ori_f0_data = io_funcs.load_float_file(f0_file) ## to process float f0 file
                interp_f0_arr = prosody_funcs.interpolate_f0(f0_file)
            
                ### read label file ###
                [phone, ph_arr, mean_f0_arr] = prosody_funcs.read_state_align_label_file(lab_file)
                
                ### decomposition of f0 file ###
                [dct_features, stat_features] = prosody_funcs.DCT_decomposition_from_lab_file(phone, ph_arr, mean_f0_arr, interp_f0_arr,
                                                                                                decomposition_unit, coef_size, stat_size)
                
                flens.append(len(dct_features) + len(stat_features))
                for j in range(len(dct_features) / coef_size):
                    prosody_feats = np.concatenate((prosody_feats, dct_features[j * coef_size:(j + 1) * coef_size]), axis=0)
                    prosody_feats = np.concatenate((prosody_feats, stat_features[j * stat_size:(j + 1) * stat_size]), axis=0)
            
            
            ##### normalise the data #####
            print 'Normalising the data....'
            if(normalization == "MVN"):
                norm_data = data_normalization.MVN_normalize(prosody_feats, out_dim, stats_file)
            else:
                norm_data = prosody_feats
            
                    
            ##### write features into files ####
            print 'Writing features into output files...'
            count = 0;idx = 0;flength = 0;
            for k in range(len(list_arr)):
                filename = list_arr[k]        
                print filename
                out_file = os.path.join(out_dir, filename + '.cmp')
                op1 = open(out_file, 'w');
                flength = flength + flens[k]
                while idx < flength:
                    op1.write(str(norm_data[idx]) + '\n')
                    idx = idx + 1
                op1.close();

                # norm_file_data = norm_data[count:count+flens[k]]
                # io_funcs.array_to_binary_file(norm_file_data, out_file)
                # count+=flens[k]
                # ## comment below line to run full list of files
                # break; ### breaks after processing one file - to check errors
            
        recons_f0_data_all_files = []
        ori_f0_data_all_files = []
        base_f0_data_all_files = []
        interp_f0_data_all_files = []        
        
        if prosodyrecons:
            gen_dir = os.path.join(work_dir, '../Feature_extraction/Output/gen/blzpilot/f0_contour_dct/')          
            # dnn_gen_dir = 'DNN__contour_1_1900_'+str(in_dim)+'_'+str(out_dim)+'_6_256'
            # gen_dir     = os.path.join(work_dir,'../dnn_tts_contour/two_stage_mtldnn/gen/',dnn_gen_dir)
            base_f0_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/baseline-f0/without_sil/float/')
            stats_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', stat_fname)
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot_test.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            remove_silence = True
            apply_base_vuv_on_pred_f0 = True
            
            for k in range(len(list_arr)):
                filename = list_arr[k]
                print filename
                lab_file = os.path.join(lab_dir, filename + '.lab')
                f0_file = os.path.join(f0_dir, filename + '.f0')
                gen_file = os.path.join(gen_dir, filename + '.cmp')
                base_f0_file = os.path.join(base_f0_dir, filename + '.lf0')
                
                ### processing original f0 file ###
                # ori_lf0_data, frame_number = io_funcs.load_binary_file_frame(lf0_file, 1)
                ori_f0_data = io_funcs.load_float_file(f0_file)  # # to process float f0 file
                base_f0_data = io_funcs.load_float_file(base_f0_file)  # # to process float f0 file
                interp_f0_data = prosody_funcs.interpolate_f0(f0_file)
                
                ### read label file ###
                [phone, ph_arr, mean_f0_arr] = prosody_funcs.read_state_align_label_file(lab_file)
                
                ### load generated output ###
                # gen_data = io_funcs.load_binary_file(gen_file, 1)
                gen_data = io_funcs.load_float_file(gen_file)
                                
                ##### denormalization of data #####
                print 'denormalizing the data....'
                if(normalization == "MVN"):
                    denorm_data = data_normalization.MVN_denormalize(gen_data, out_dim, stats_file)
                else:
                    denorm_data = gen_data          
                
                dct_features = []; stat_features = []
                for j in range(len(denorm_data) / out_dim):
                    dct_features = np.concatenate((dct_features, denorm_data[j * out_dim:((j) * out_dim + coef_size)]), axis=0)
                    stat_features = np.concatenate((stat_features, denorm_data[j * out_dim + coef_size:(j + 1) * out_dim]), axis=0)
                
                recons_f0_contour = prosody_funcs.DCT_reconstruction_from_lab_file(phone, ph_arr, mean_f0_arr, interp_f0_data, 
                                                                                   denorm_data, decomposition_unit, coef_size, out_dim, True)
                if remove_silence:
                    ph_start = int(ph_arr[0][1] / (np.power(10, 4) * 5));
                    ph_end = int(ph_arr[1][len(phone) - 2] / (np.power(10, 4) * 5));
                    ori_f0_data = ori_f0_data[ph_start:ph_end]
                    interp_f0_data = interp_f0_data[ph_start:ph_end]
                
                if apply_base_vuv_on_pred_f0:
                    for x in range(len(base_f0_data)):
                        if(base_f0_data[x] == 0):
                            recons_f0_contour[x] = 0
                        if(ori_f0_data[x] == 0):
                            interp_f0_data[x] = 0
                
                #prosody_funcs.plot_DBR(interp_f0_data, base_f0_data)
                #prosody_funcs.plot_DBR(interp_f0_data, recons_f0_contour)
                    
                if(len(ori_f0_data) > len(recons_f0_contour)):
                    extralen = len(ori_f0_data) - len(recons_f0_contour)
                    extra_zeros = prosody_funcs.zeros(extralen, 1)
                    recons_f0_contour = np.concatenate((recons_f0_contour, extra_zeros), axis=0)
                elif(len(ori_f0_data) > len(recons_f0_contour)):
                    recons_f0_contour = recons_f0_contour[0:len(ori_f0_data)]
                
                recons_f0_data_all_files = np.concatenate((recons_f0_data_all_files, recons_f0_contour), axis=0)
                ori_f0_data_all_files = np.concatenate((ori_f0_data_all_files, ori_f0_data), axis=0)
                base_f0_data_all_files = np.concatenate((base_f0_data_all_files, base_f0_data), axis=0)
                interp_f0_data_all_files = np.concatenate((interp_f0_data_all_files, interp_f0_data), axis=0)
                # break;
                      
        if computermse:    
            ### evaluation metrics ###
            rmse_error, vuv_error = eval_metrics.rmse_with_vuv(base_f0_data_all_files, ori_f0_data_all_files)
            print 'F0: ' + str(rmse_error) + ' Hz; VUV: ' + str(vuv_error) + '%'
            print 'CORR: ' + str(eval_metrics.corr_with_vuv(base_f0_data_all_files, ori_f0_data_all_files))
            
            ### evaluation metrics ###
            rmse_error, vuv_error = eval_metrics.rmse_with_vuv(recons_f0_data_all_files, ori_f0_data_all_files)
            print 'F0: ' + str(rmse_error) + ' Hz; VUV: ' + str(vuv_error) + '%'
            print 'CORR: ' + str(eval_metrics.corr_with_vuv(recons_f0_data_all_files, ori_f0_data_all_files))
            
