'''
Created on 9 Nov 2015

@author: Srikanth Ronanki
'''
import os
from io_funcs.binary_io import BinaryIOCollection
from io_funcs.read_hts_label_file import readHTSlabelFile
import numpy as np

class linguistic_features():
    
    def __init__(self, vowelListPath):
        ip1 = open(vowelListPath,'r')
        self.vlist = [x.strip() for x in ip1.readlines()]
        ip1.close() 
        pass
    
    def load_word_embeddings(self, word_embed_file):
        self.wrd_embeds = {}
        ip1 = open(word_embed_file, 'r')
        for i in ip1.readlines():
            fstr = i.strip().split()
            word_vec = ' '.join(map(str, fstr[1:]))
            self.wrd_embeds[fstr[0]] = word_vec
        ip1.close()
        
    def extract_base_features(self, feat_dir_path, feat_switch, list_of_files, decomposition_unit, unit_dim):
        ### load Binary module ###
        io_funcs = BinaryIOCollection()
        htsclass = readHTSlabelFile()
        
        max_vow_syl=''
        max_vow_index = [];
        max_vow_utt=''
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
            
            word_embed_list = []
            binary_feat_list = []
            identity_vec_list = []
            
            wc = 0; phinsyl=0;
            syl_identity = self.zeros(300,1)
            syl = ''
            
            j=0
            while j < file_len: 
                ### get phone-level indices ###
                st_ph_indx = int(mean_f0_arr[j][0]);
                fn_ph_indx = int(mean_f0_arr[j][1]);
                    
                ### for silence and pause regions ###    
                if(st_ph_indx == 0 or fn_ph_indx == 0):
                    st_ph_indx = j
                    fn_ph_indx = j
    
                ph_start = int(ph_arr[0][st_ph_indx] / (np.power(10, 4) * 5));
                ph_end = int(ph_arr[1][fn_ph_indx] / (np.power(10, 4) * 5));
                mid_frame = (ph_start+ph_end)/2
                
                #### take zeros for silence ####
                if(phone[j] == '#'):
                    j = j + 1
                    continue;
                
                st_indx = unit_dim['frame']*mid_frame
                frame_feat = features[st_indx:st_indx+592]
                
                ph_identity = frame_feat[99:148]
                ph_identity = np.reshape(ph_identity, len(ph_identity), -1)
                st_indx = phinsyl*50
                syl_identity[st_indx:st_indx+49] = ph_identity
                syl = syl + phone[j]
                if phone[j] in self.vlist:
                    vow_index = phinsyl
                if phone[j] == '#':
                    syl_identity[(phinsyl+1)*50-1] = 1
                phinsyl += 1
                
                ### syllable ending information ###
                syl_end = 0        
                if(mean_f0_arr[j + 1][3] - mean_f0_arr[j][3] != 0):
                    syl_end = 1
                
                ### word ending information ###
                word_end = 0        
                if(mean_f0_arr[j + 1][5] - mean_f0_arr[j][5] != 0):
                    word_end = 1
                
                if(decomposition_unit=='syllable' and syl_end):
                    
                    ### binary feature set for syllable ###
                    syl_feat = []
                    for x in range(len(frame_feat)):
                        if(x < 348 or (x >= 405 and x < 421)):
                            continue;
                        syl_feat.append(frame_feat[x])
                    norm_syl_feat = [0.99 if x==1 else 0.01 for x in syl_feat]
                    norm_syl_vec = ' '.join(map(str, norm_syl_feat[:]))
                    binary_feat_list.append(norm_syl_vec)
                    
                    ### word embeddings for syllable ###
                    word = list_of_words[wc]
                    if(word_end and phone[j]!='pau'): 
                        wc += 1    
                    if(phone[j] == 'pau'):
                        word_vec = self.wrd_embeds['*UNKNOWN*']
                    elif word in self.wrd_embeds:
                        word_vec = self.wrd_embeds[word]
                    else:
                        word_vec = self.wrd_embeds['*UNKNOWN*']        
                    word_embed_list.append(word_vec)
                    
                    ### identity features for syllable ###
                    #if(vow_index<=1):
                    #    syl_identity = np.roll(syl_identity, 50*(vow_index+1))
                    norm_syl_identity = [0.99 if x==1 else 0.01 for x in syl_identity]
                    norm_syl_identity_vec = ' '.join(map(str, norm_syl_identity[:]))
                    identity_vec_list.append(norm_syl_identity_vec)
                    ### reset syllable information ###
                    #print syl
                    phinsyl = 0; syl=''
                    syl_identity = self.zeros(300, 1)    
                
                j+=1                   
            
            syl_identity = self.zeros(300, 1)
            norm_syl_identity = [0.99 if x==1 else 0.01 for x in syl_identity]
            norm_syl_identity_vec = ' '.join(map(str, norm_syl_identity[:]))
            word_vec = self.wrd_embeds['*UNKNOWN*']
            
            op1 = open(out_filename, 'w')
            for x in range(len(binary_feat_list)):
                if feat_switch['binary']:
                    op1.write(binary_feat_list[x]+' ')
                #if(x-1<0):
                #    op1.write(word_vec+' ')
                #else:
                #    op1.write(word_embed_list[x-1]+' ')
                if feat_switch['wordEmbed']:
                    op1.write(word_embed_list[x]+' ')
                #if(x+1>=len(binary_feat_list)):
                #    op1.write(word_vec+' ')
                #else:
                #    op1.write(word_embed_list[x+1]+' ')
                #if(x-2<0):
                #    op1.write(norm_syl_identity_vec+' ')
                #else:
                #    op1.write(identity_vec_list[x-2]+' ')
                #if(x-1<0):
                #    op1.write(norm_syl_identity_vec+' ')
                #else:
                #    op1.write(identity_vec_list[x-1]+' ')
                if feat_switch['identity']:
                    op1.write(identity_vec_list[x]+' ')
                #if(x+1>=len(binary_feat_list)):
                #    op1.write(norm_syl_identity_vec+' ')
                #else:
                #    op1.write(identity_vec_list[x+1]+' ')
                #if(x+2>=len(binary_feat_list)):
                #    op1.write(norm_syl_identity_vec+' ')
                #else:
                #    op1.write(identity_vec_list[x+2]+' ')
                op1.write('\n')
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
    
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model'
    dnn_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/dnn_tts_blzpilot/'
    
    decomposition_unit = 'syllable'
    word_embed_size = 50
    identity_size = 300
    baseline_size = 228
    
    unit_dim = {}
    unit_dim['frame']    = 601
    unit_dim['phone']    = 592
    unit_dim['syllable'] = 228
    unit_dim['word']     = 92
    
    in_dim  = 601
    out_dim = 300
 
    binary_label_dir = os.path.join(dnn_dir, 'two_stage_mtldnn/data/binary_label_' + str(in_dim))
    label_align_dir = os.path.join(dnn_dir, 'two_stage_mtldnn/data/label_state_align')  
    text_dir = os.path.join(work_dir, 'Data/database/blzpilot/txt/')
    
    #out_dir = decomposition_unit+'_baseline_'+str(baseline_size)+'_wordembed_'+str(word_embed_size)+'_identity_'+str(identity_size)
    out_dir = decomposition_unit+'_identity_'+str(identity_size)
    out_feat_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/label_features/' + str(out_dir) + '/binary_label_' + str(out_dim))
    if not os.path.exists(out_feat_dir):
        os.makedirs(out_feat_dir)
        
    feat_switch = {}
    feat_switch['binary']     = 0
    feat_switch['wordEmbed']  = 0
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
    
        filelist = os.path.join(work_dir, 'Data/fileList/blzpilot.scp')
        word_embed_file = os.path.join(work_dir, 'Data/word_embeddings/turian-embeddings-50.txt')
        vowelListPath = os.path.join(work_dir, 'Data/phoneset/vowels.txt')
        
        ip_feats = linguistic_features(vowelListPath)
        io_funcs = BinaryIOCollection()
        
        if extract_base_feats: 
            list_of_files = io_funcs.load_file_list(filelist)
            ip_feats.load_word_embeddings(word_embed_file)
            ip_feats.extract_base_features(feat_dir_path, feat_switch, list_of_files, decomposition_unit, unit_dim)
        
