'''
Created on 9 Nov 2015

@author: Srikanth Ronanki
'''
import os
from io_funcs.binary_io import BinaryIOCollection
from io_funcs.read_hts_label_file import readHTSlabelFile
import numpy as np

class linguistic_features(object):
    
    def __init__(self):
        self.plist = []
        pass
    
    def load_word_embeddings(self, word_embed_file):
        self.wrd_embeds = {}
        ip1 = open(word_embed_file, 'r')
        for i in ip1.readlines():
            fstr = i.strip().split()
            for j in range(len(fstr)):
                if(j == 0):
                    word_vec = ''
                    continue
                else:
                    word_vec = word_vec + fstr[j] + ' '
            self.wrd_embeds[fstr[0]] = word_vec[0:len(word_vec) - 1]
        ip1.close()
        
    def extract_base_features(self, feat_dir_path, feat_switch, list_of_files, decomposition_unit):
        ### load Binary module ###
        io_funcs = BinaryIOCollection()
        htsclass = readHTSlabelFile()
        
        ### read file by file ###
        for i in range(len(list_of_files)):    
            filename = list_of_files[i]     
            print filename
            
            binary_label_dir = feat_dir_path['input_binary']
            label_align_dir = feat_dir_path['input_labfile']
            txt_dir = feat_dir_path['input_txt']
            out_feat_dir = feat_dir_path['output_feat']
            
            in_filename = os.path.join(binary_label_dir, filename + '.lab');
            in_lab_file = os.path.join(label_align_dir, filename + '.lab')
            in_txt_file = os.path.join(txt_dir, filename + '.txt')
            out_filename = os.path.join(out_feat_dir, filename + '.lab');
            
            ip1 = open(in_txt_file, 'r')
            text_Data = ip1.readlines()
            ip1.close()
            
            list_of_words = text_Data[0].split()

            [phone, ph_arr, mean_f0_arr] = htsclass.read_state_align_label_file(in_lab_file)
            
            features = io_funcs.load_binary_file(in_filename, 1)
        
            file_len = len(phone)
            
            op1 = open(out_filename, 'w')
            count = 0; frame_count = 0; phone_count = 0;
            wc = 0; seg_count = 0;
        
            feat_arr = []
            prev_feat_arr = []
            syl_identity = self.zeros(300,1)
            syl = ''
            phinsyl = 0
            for j in range(len(features)):
                count = count + 1
        
                if(count == 601):
                    count = 0;
                    feat_arr = []
                    sil_flag = 0
                    continue;
        
                if (count == 59 and int(features[j]) == 1):
                    sil_flag = 1
                if (count == 148 and int(features[j]) == 1):
                    sil_flag = 0
        
                if(count <= 348 or (count >= 406 and count <= 421) or count > 592):
                    continue;
        
                feat_arr.append(int(features[j]))
        
                if(count == 592):
                    
                    if np.abs(frame_count - int(ph_arr[1][phone_count] * (10 ** -4) / 5)) <= 1:
                            ph_identity = features[j-492:j-443]
                            ph_identity = np.reshape(ph_identity, len(ph_identity), -1)
                            syl_identity[phinsyl*50:(phinsyl+1)*50-1] = ph_identity
                            syl = syl+phone[phone_count]
                            if phone[phone_count] == '#':
                                syl_identity[(phinsyl+1)*50-1] = 1
                            phinsyl += 1
                            phone_count += 1
                    
                    frame_count += 1
          
                    if(len(prev_feat_arr) != 0 and prev_feat_arr == feat_arr):
                        continue;
                    else:
                        prev_feat_arr = feat_arr
                        
                        if(syl!='#' and syl!=''):
                            syl_vec = ''
                            new_syl_identity = [0.99 if x==1 else 0.01 for x in syl_identity]
                            for x in range(len(new_syl_identity)):
                                syl_vec = syl_vec+str(new_syl_identity[x])+' '
                            op1.write(syl_vec+'\n')
                        
                        ### reset syllable information ###
                        phinsyl = 0; syl=''
                        syl_identity = self.zeros(300, 1)
                        
                        if (sil_flag == 1):
                            continue;
                        seg_count += 1
                        new_arr = [0.99 if x==1 else 0.01 for x in prev_feat_arr]
                        for item in new_arr:
                            op1.write("%s " % item)
                                
                        ### word ending information ###        
                        if(mean_f0_arr[phone_count][5] - mean_f0_arr[phone_count - 1][5] != 0 and phone[phone_count] != 'pau'):
                            wc += 1
                        word = list_of_words[wc - 1]    
                        if word in self.wrd_embeds:
                            word_vec = self.wrd_embeds[word]
                        else:
                            word_vec = self.wrd_embeds['*UNKNOWN*']
                        if(phone[phone_count] == 'pau'):
                            word_vec = self.wrd_embeds['*UNKNOWN*']
                        op1.write(word_vec + ' ')
                        continue;
            op1.close()
            #break;
    
    def zeros(self, m, n):
        if(n == 1):
            arr = np.ndarray((m,), float)
        else:
            arr = np.ndarray((m, n), float)
        arr.fill(0)
        return arr

if __name__ == "__main__":
    
    ip_feats = linguistic_features()
    io_funcs = BinaryIOCollection()
    
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model'
    dnn_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/dnn_tts_blzpilot/'
    
    decomposition_unit = 'syllable'
    word_embed_size = 50
    
    unit_dim = {}
    unit_dim['frame']    = 601
    unit_dim['phone']    = 592
    unit_dim['syllable'] = 228
    unit_dim['word']     = 92
    
    in_dim  = 601
    out_dim = unit_dim[decomposition_unit] + word_embed_size
 
    binary_label_dir = os.path.join(dnn_dir, 'two_stage_mtldnn/data/binary_label_' + str(in_dim))
    label_align_dir = os.path.join(dnn_dir, 'two_stage_mtldnn/data/label_state_align')  
    text_dir = os.path.join(work_dir, 'Data/database/blzpilot/txt/')
    out_feat_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/label_features/' + str(decomposition_unit) + '_baseline_228/binary_label_' + str(out_dim))
    if not os.path.exists(out_feat_dir):
        os.makedirs(out_feat_dir)
        
    feat_switch = {}
    feat_switch['binary']     = 1
    feat_switch['wordEmbed']  = 1
    feat_switch['identity']   = 1
    feat_switch['bottleneck'] = 0
    
    feat_dir_path = {}
    feat_dir_path['input_binary']  = binary_label_dir
    feat_dir_path['input_labfile'] = label_align_dir
    feat_dir_path['input_txt']     = text_dir
    feat_dir_path['output_feat']   = out_feat_dir
        
    
    DFP = 1
    if DFP:
        extract_base_feats = True;
        
        if extract_base_feats:
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot.scp')
            word_embed_file = os.path.join(work_dir, 'Data/word_embeddings/turian-embeddings-50.txt')
            list_of_files = io_funcs.load_file_list(filelist)
            ip_feats.load_word_embeddings(word_embed_file)
            ip_feats.extract_base_features(feat_dir_path, feat_switch, list_of_files, decomposition_unit)
        
