import os
import numpy as np
from run_dct import DCTFeatures
from io_funcs.binary_io import BinaryIOCollection
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import inconsistent
from scipy.cluster.hierarchy import fcluster
from utils import eval_metrics
from models.dct_models import DCTModels
from utils import data_normalization


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def plot_templates(final_clusters, mean_f0=160, num_of_frames=20):
    [k, coef_size] = final_clusters.shape
    dct_funcs = DCTModels()
    mean_f0 = np.log10(0.00437*mean_f0+1)/np.sqrt(num_of_frames)
    c0 = (1.0 * mean_f0) * float(num_of_frames) * 2 * np.sqrt(1 / (4 * float(num_of_frames)));
    for i in range(k):
        arr2 = []
        arr2.append(c0)
        for j in range(coef_size):
            arr2.append(final_clusters[i,j])
        template = dct_funcs.idct(arr2, num_of_frames)
        gen_f0 = np.power(10, template)
        gen_f0 = (np.array(gen_f0) - 1) / 0.00437; 
    
        ax = plt.figure(1).add_subplot(k/2, 2, i+1)
        ax.plot(gen_f0, label='DCT 10 cluster decomposition')
        ax.set_xlim([1, len(gen_f0)])
        #plt.legend(loc="upper right")
    
    
    #plt.title('DCT decomposition - 10 templates')
    #plt.xlabel("F0 Contour at syllable level")
    #plt.ylabel("F0 (Hz)")
    plt.show()

def find_clusters(Y, final_clusters, coef_size):
    clusters = []
    for i in range(len(Y)):
        min_dist = 1000;
        for j in range(len(final_clusters)):
            dist = np.sqrt(sum(np.square(Y[i,:] - final_clusters[j,:]))/coef_size)
            if dist<min_dist:
                min_cluster = j+1
                min_dist = dist
        clusters.append(min_cluster)
    return clusters

if __name__ == "__main__":
    ###    This is main function   ###    
    ### load all modules ###
    prosody_funcs = DCTFeatures()
    io_funcs = BinaryIOCollection()
    dct_models = DCTModels() 

    ### model parameters ###
    normalization = 'hybrid'
    decomposition_unit = 'syllable' 
    coef_size = 9
    stat_size = 10
    out_dim = coef_size + stat_size
    num_of_clusters = 6
    num_of_clusters1 = 2
    out_feat_dim = num_of_clusters
    in_dim = 592
    
    ### Relative work path ###
    # work_dir = os.path.join(os.getcwd(), "../../")
    
    ### Absolute work path ###
    work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model/'
    if out_feat_dim > num_of_clusters:
        feat_dir_path = decomposition_unit + '_template_' + str(num_of_clusters) + '_stat_' + str(out_feat_dim-num_of_clusters)
    else:
        feat_dir_path = decomposition_unit + '_template_' + str(num_of_clusters1)
    dct_dir_path = decomposition_unit + '_dct_' + str(coef_size) + '_stat_' + str(stat_size)
    f0_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/f0/')
    lab_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/label_state_align/')
    out_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/template_features/' + feat_dir_path + '/')
    out_f0_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/gen/lf0/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_f0_dir):
        os.makedirs(out_f0_dir)
    
    ### Single file processing ###
    SFP = True;
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
        
        nrows = len(dct_features)/coef_size
        X1 = np.array(dct_features).reshape(nrows, coef_size)
        
        Y = X1[:, 1:]
        
        Z = linkage(Y, 'ward')

        print Y.shape
        print Z.shape
        
        '''
        max_d = 50
        clusters = fcluster(Z, max_d, criterion='distance')
        print clusters
        '''
        
        k=num_of_clusters
        clusters = fcluster(Z, k, criterion='maxclust')
        print clusters 
        max_d = 0.250
        fancy_dendrogram(
            Z,
            truncate_mode='lastp',
            p=k,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            annotate_above=10,
            max_d=max_d,  # plot a horizontal cut-off line
        )
        plt.show()
        
        clusters_contours = []
        for xi in range(k):
            ind_cnt = np.where(clusters == xi+1)
            clusters_contours = np.concatenate((clusters_contours, np.mean(Y[ind_cnt], axis=0)), axis=0)
        
        final_clusters = clusters_contours.reshape(k,coef_size-1)
        print final_clusters
        for xi in range(len(clusters)):
            cluster_indx = clusters[xi] - 1
            X1[xi, 1:] = final_clusters[cluster_indx, :]
            
        template_dct_features = X1.reshape(len(dct_features),)
            
        ### reconstruction of f0 file ###
        recons_f0_contour = prosody_funcs.DCT_reconstruction(template_dct_features, stat_features, coef_size)
        
        ### evaluation metrics ###
        print 'RMSE: ' + str(eval_metrics.rmse(recons_f0_contour, interp_f0_arr[0:len(recons_f0_contour)]))
        print 'CORR: ' + str(eval_metrics.corr(recons_f0_contour, interp_f0_arr[0:len(recons_f0_contour)]))
        
        ### plot ###
        #prosody_funcs.plot_dct(dct_features)       
        prosody_funcs.plot_DBR(interp_f0_arr, recons_f0_contour)

### Directory of files processing ###
    DFP = True;
    if DFP:
        prosodydecomp = False;
        hierarcluster = False;
        templatefeats = False;
        prosodyrecons = True;
        computermse = True;
        analyseresults = True;
        
        if prosodydecomp:
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot_train.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            prosody_feats = []; flens = [];
            dct_features_all_files = []
            stat_features_all_files = []
            for k in range(350):
                #if(k<1180):
                #    continue
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
                
                dct_features_all_files = np.concatenate((dct_features_all_files, dct_features), axis=0)
                stat_features_all_files = np.concatenate((stat_features_all_files, stat_features), axis=0)             
                        
        if hierarcluster:
            clusters_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', 'final_clusters_'+str(num_of_clusters)+'.txt')
            nrows = len(dct_features_all_files)/coef_size
            X1 = np.array(dct_features_all_files).reshape(nrows, coef_size)
            
            Y = X1[:,1:]
            Z = linkage(Y, 'ward')
               
            k=num_of_clusters
            clusters = fcluster(Z, k, criterion='maxclust')
            
            max_d = 1.250
            fancy_dendrogram(
                Z,
                truncate_mode='lastp',
                p=k,
                leaf_rotation=90.,
                leaf_font_size=12.,
                show_contracted=True,
                annotate_above=10,
                max_d=max_d,  # plot a horizontal cut-off line
            )
            plt.show()
            
            
            clusters_contours = []
            for xi in range(k):
                ind_cnt = np.where(clusters == xi+1)
                clusters_contours = np.concatenate((clusters_contours, np.mean(Y[ind_cnt], axis=0)), axis=0)
            
            final_clusters = clusters_contours.reshape(k,coef_size-1)
            
            if k%2==0:
                plot_templates(final_clusters)
            
            io_funcs.array_to_binary_file(final_clusters, clusters_file)   
            # ## comment below line to run full list of files
            # break; ### breaks after processing one file - to check errors
        
        train_clusters = []
        dev_clusters   = []
        test_clusters = []
        if templatefeats:
            stat_fname = feat_dir_path + '.txt'
            stats_template_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', stat_fname)
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            prosody_feats = []; flens = [];
            template_class_features = []
            for k in range(len(list_arr)):
                #if(k<1180):
                #    continue
                filename = list_arr[k]
                out_file = os.path.join(out_dir, filename + '.cmp')
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
                
                nrows = len(dct_features)/coef_size
                X1 = np.array(dct_features).reshape(nrows, coef_size)
                Y = X1[:,1:]
                
                clusters = find_clusters(Y, final_clusters, coef_size)
                for xi in range(len(clusters)):
                    cluster_ = clusters[xi]
                    if k<1900:
                        train_clusters.append(cluster_)  
                    elif k<1992:
                        dev_clusters.append(cluster_)
                    else:
                        test_clusters.append(cluster_)   
                    fzeros = prosody_funcs.zeros(out_feat_dim, 1)
                    fzeros[cluster_ - 1] = 1
                    num_of_frames = stat_features[((xi + 1) * stat_size) - 1]
                    mean_f0 = stat_features[(xi * stat_size) + 1]
                    if out_feat_dim>10:
                        fzeros[-2] = mean_f0
                        fzeros[-1] = num_of_frames
                    template_class_features = np.concatenate((template_class_features, fzeros), axis=0)
                    
                flens.append(len(clusters)*out_feat_dim)
                #io_funcs.array_to_binary_file(template_class_features, out_file)
                #break  
            
            ##### normalise the data #####
            print 'Normalising the data....'
            normalization = "no"
            if(normalization == "hybrid"):
                norm_data = data_normalization.hybrid_normalize(template_class_features, out_feat_dim, stats_template_file)
            else:
                norm_data = template_class_features
            
                    
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
                    '''
                    if(norm_data[idx]==1):
                        rem_idx = idx % num_of_clusters
                        if(rem_idx==4):
                            op1.write('0\n1\n')
                        else:
                            op1.write('1\n0\n')
                    '''        
                    op1.write(str(norm_data[idx]) + '\n')
                    idx = idx + 1
                op1.close();
                #break

                # norm_file_data = norm_data[count:count+flens[k]]
                # io_funcs.array_to_binary_file(norm_file_data, out_file)
                # count+=flens[k]
                # ## comment below line to run full list of files
                # break; ### breaks after processing one file - to check errors
      
            
        recons_f0_data_all_files = []
        ori_f0_data_all_files = []
        base_f0_data_all_files = []
        interp_f0_data_all_files = []
        pred_clusters = []
        hit_rate = 0; total_syl = 0        
        
        if prosodyrecons:
            stat_template_fname = feat_dir_path + '.txt'
            stat_fname = dct_dir_path + '.txt'
            gen_dir = os.path.join(work_dir, '../Feature_extraction/Output/gen/blzpilot/f0_contour_dct/')
            gen_template_dir = os.path.join(work_dir, '../Feature_extraction/Output/gen/blzpilot/cmp/')          
            # dnn_gen_dir = 'DNN__contour_1_1900_'+str(in_dim)+'_'+str(out_dim)+'_6_256'
            # gen_dir     = os.path.join(work_dir,'../dnn_tts_contour/two_stage_mtldnn/gen/',dnn_gen_dir)
            base_f0_dir = os.path.join(work_dir, 'Data/inter-module/blzpilot/baseline-f0/without_sil/float/')
            stats_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', stat_fname)
            stats_template_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', stat_template_fname)
            clusters_file = os.path.join(work_dir, 'Data/inter-module/blzpilot/misc/', 'final_clusters_'+str(num_of_clusters)+'.txt')
            filelist = os.path.join(work_dir, 'Data/fileList/blzpilot_test.scp')
            list_arr = io_funcs.load_file_list(filelist)
            
            remove_silence = True
            apply_base_vuv_on_pred_f0 = True
            apply_org_vuv_on_pred_f0  = False
            write_output_to_file = False
            load_gen_data = True
            load_gen_template_data = True
            
            final_clusters, frame_number = io_funcs.load_binary_file_frame(clusters_file, coef_size-1)
            #plot_templates(final_clusters)
            
            for k in range(len(list_arr)):
                filename = list_arr[k]
                print filename
                lab_file = os.path.join(lab_dir, filename + '.lab')
                f0_file = os.path.join(f0_dir, filename + '.f0')
                gen_file = os.path.join(gen_dir, filename + '.cmp')
                gen_template_file = os.path.join(gen_template_dir, filename + '.cmp')
                base_f0_file = os.path.join(base_f0_dir, filename + '.lf0')
                output_f0_file = os.path.join(out_f0_dir, filename+ '.lf0')
                
                ### processing original f0 file ###
                # ori_lf0_data, frame_number = io_funcs.load_binary_file_frame(lf0_file, 1)
                ori_f0_data = io_funcs.load_float_file(f0_file)  # # to process float f0 file
                base_f0_data = io_funcs.load_float_file(base_f0_file)  # # to process float f0 file
                interp_f0_data = prosody_funcs.interpolate_f0(f0_file)
                
                ### read label file ###
                [phone, ph_arr, mean_f0_arr] = prosody_funcs.read_state_align_label_file(lab_file)
                
                ### original length ###
                f0_org_len = len(interp_f0_data)
                
                if load_gen_data:
                    ### load generated output ###
                    #gen_data = io_funcs.load_binary_file(gen_file, 1)
                    gen_data = io_funcs.load_float_file(gen_file)
                    
                    ##### denormalization of data #####
                    #print 'denormalizing the data....'
                    normalization = "MVN"
                    if(normalization == "MVN"):
                        denorm_data = data_normalization.MVN_denormalize(gen_data, out_dim, stats_file)
                    else:
                        denorm_data = gen_data          
                    
                    dct_features = []; stat_features = []
                    for j in range(len(denorm_data) / out_dim):
                        dct_features = np.concatenate((dct_features, denorm_data[j * out_dim:((j) * out_dim + coef_size)]), axis=0)
                        stat_features = np.concatenate((stat_features, denorm_data[j * out_dim + coef_size:(j + 1) * out_dim]), axis=0)
                
                if load_gen_template_data:
                    ### load generated output ###
                    #gen_data = io_funcs.load_binary_file(gen_file, 1)
                    gen_data = io_funcs.load_float_file(gen_template_file)
                    
                    ##### denormalization of data #####
                    #print 'denormalizing the data....'
                    normalization = "no"
                    if(normalization == "hybrid"):
                        denorm_data = data_normalization.hybrid_denormalize(gen_data, out_feat_dim, stats_template_file)
                    else:
                        denorm_data = gen_data          
                    
                    clusters = []
                    for j in range(len(denorm_data) / out_feat_dim):
                        cluster_vector = denorm_data[j * out_feat_dim : j * out_feat_dim + num_of_clusters]
                        cluster_class = cluster_vector.index(max(cluster_vector))
                        clusters.append(cluster_class+1)
                
                ### original features ###      
                [org_dct_features, org_stat_features] = prosody_funcs.DCT_decomposition_from_lab_file(phone, ph_arr, mean_f0_arr, interp_f0_data,
                                                                                                decomposition_unit, coef_size, stat_size)
                
                #org_dct_features = dct_features
                org_stat_features = stat_features          
                
                nrows = len(org_dct_features)/coef_size
                X1 = np.array(org_dct_features).reshape(nrows, coef_size)
                Y = X1[:,1:]
                
                
                cc_clusters = find_clusters(Y, final_clusters, coef_size)
                
                for xi in range(len(clusters)):
                    test_clusters.append(cc_clusters[xi])
                    pred_clusters.append(clusters[xi])
                    cluster_indx = cc_clusters[xi] - 1
                    X1[xi, 1:] = final_clusters[cluster_indx, :]
                    ### accuracy measure ###
                    '''
                    if(cc_clusters[xi]==5):
                        cc_clusters[xi]=2
                    else:
                        cc_clusters[xi]=1
                    '''    
                    if(cc_clusters[xi]==clusters[xi]):
                        hit_rate+=1
                
                total_syl += len(clusters)
                template_dct_features = X1.reshape(len(org_dct_features),)
                
                ### reconstruction of f0 file ###
                denorm_data = []
                for j in range(len(org_dct_features) / coef_size):
                    denorm_data = np.concatenate((denorm_data, template_dct_features[j * coef_size:(j + 1) * coef_size]), axis=0)
                    denorm_data = np.concatenate((denorm_data, org_stat_features[j * stat_size:(j + 1) * stat_size]), axis=0)
                
                recons_f0_contour = prosody_funcs.DCT_reconstruction_from_lab_file(phone, ph_arr, mean_f0_arr, interp_f0_data, 
                                                                                   denorm_data, decomposition_unit, coef_size, out_dim, True)
                #recons_f0_contour  = prosody_funcs.DCT_reconstruction(template_dct_features, org_stat_features, coef_size)
                
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
                
                if apply_org_vuv_on_pred_f0:
                    for x in range(len(base_f0_data)):
                        if(ori_f0_data[x] == 0):
                            interp_f0_data[x] = 0
                            recons_f0_contour[x] = 0
                
                if write_output_to_file:
                    f0_data_with_sil=[]
                    zero_arr = prosody_funcs.zeros(ph_start, 1)
                    f0_data_with_sil = zero_arr
                    f0_data_with_sil = np.concatenate((f0_data_with_sil, base_f0_data), axis=0)
                    zero_arr = prosody_funcs.zeros(f0_org_len - ph_end, 1)
                    f0_data_with_sil = np.concatenate((f0_data_with_sil, zero_arr), axis=0)
                    io_funcs.array_to_binary_file(f0_data_with_sil, output_f0_file)
                    
                #prosody_funcs.plot_DBR(interp_f0_data, base_f0_data)
                #prosody_funcs.plot_DBR(interp_f0_data, recons_f0_contour)
                
                '''    
                if(len(ori_f0_data) > len(recons_f0_contour)):
                    extralen = len(ori_f0_data) - len(recons_f0_contour)
                    extra_zeros = prosody_funcs.zeros(extralen, 1)
                    recons_f0_contour = np.concatenate((recons_f0_contour, extra_zeros), axis=0)
                elif(len(ori_f0_data) > len(recons_f0_contour)):
                    recons_f0_contour = recons_f0_contour[0:len(ori_f0_data)]
                '''
                
                ori_f0_data_all_files = np.concatenate((ori_f0_data_all_files, ori_f0_data), axis=0)
                base_f0_data_all_files = np.concatenate((base_f0_data_all_files, base_f0_data), axis=0)
                interp_f0_data_all_files = np.concatenate((interp_f0_data_all_files, interp_f0_data), axis=0)
                recons_f0_data_all_files = np.concatenate((recons_f0_data_all_files, recons_f0_contour), axis=0)
                      
        if computermse:    
            
            ### evaluation metrics ###
            rmse_error, vuv_error = eval_metrics.rmse_with_vuv(base_f0_data_all_files, ori_f0_data_all_files)
            print 'F0: ' + str(rmse_error) + ' Hz; VUV: ' + str(vuv_error) + '%'
            print 'CORR: ' + str(eval_metrics.corr_with_vuv(base_f0_data_all_files, ori_f0_data_all_files))
            
            
            #print len(recons_f0_data_all_files)
            #print len(interp_f0_data_all_files)
            #print 'RMSE: ' + str(eval_metrics.rmse(recons_f0_data_all_files, interp_f0_data_all_files))
            #print 'CORR: ' + str(eval_metrics.corr(recons_f0_data_all_files, interp_f0_data_all_files))
        
            
            ### evaluation metrics ###
            rmse_error, vuv_error = eval_metrics.rmse_with_vuv(recons_f0_data_all_files, ori_f0_data_all_files)
            print 'F0: ' + str(rmse_error) + ' Hz; VUV: ' + str(vuv_error) + '%'
            print 'CORR: ' + str(eval_metrics.corr_with_vuv(recons_f0_data_all_files, ori_f0_data_all_files))
            
            print 'No. of hits: '+str(hit_rate)
            print 'Total no. of syllables: '+str(total_syl)
            hit_percentage = float(hit_rate)/float(total_syl)*100
            print 'Class prediction accuracy: '+ str(hit_percentage) + '%'
        
        if analyseresults:
            fstr0=''; fstr1=''; fstr2=''; fstr3='';
            for xi in range(num_of_clusters):
                #ind_cnt = np.where(np.array(train_clusters) == xi+1)
                #fstr0 = fstr0+str(len(ind_cnt[0]))+' '
                #ind_cnt = np.where(np.array(dev_clusters) == xi+1)
                #fstr1 = fstr1+str(len(ind_cnt[0]))+' '
                ind_cnt = np.where(np.array(test_clusters) == xi+1)
                fstr2 = fstr2+str(len(ind_cnt[0]))+' '
                ind_cnt = np.where(np.array(pred_clusters) == xi+1)    
                fstr3 = fstr3+str(len(ind_cnt[0]))+' '
            #print fstr0
            #print fstr1
            print fstr2
            print fstr3