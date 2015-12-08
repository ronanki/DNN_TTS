'''
Created on 9 Oct 2015

@author: Srikanth Ronanki
'''
import numpy as np
from scipy.stats.stats import pearsonr

def rmse_with_vuv(predicted_arr, reference_arr):
    ### check if lengths are equal or not ###
    if(len(predicted_arr)!=len(reference_arr)):
        return 'Lengths of both arrays are not equal !!'
    else:
        arrlen = len(predicted_arr)
    diff_err = 0.0
    vuv_mismatch = 0.0
    voiced_count=0.0
    for k in range(arrlen):
        if(predicted_arr[k]>0 and reference_arr[k]>0):
            voiced_count+=1
            diff_err = diff_err + np.power((predicted_arr[k] - reference_arr[k]),2);
        elif(predicted_arr[k]==0 and reference_arr[k]==0):
            continue;
        else:
            vuv_mismatch+=1
    
    vuv_error = vuv_mismatch/float(arrlen)*100
    vuv_error = format(vuv_error, '.3f')
    diff_err  = np.sqrt(diff_err/voiced_count)
    diff_err  = format(diff_err, '.3f')
    return diff_err, vuv_error

def corr_with_vuv(predicted_arr, reference_arr):
    ### check if lengths are equal or not ###
    if(len(predicted_arr)!=len(reference_arr)):
        return 'Lengths of both arrays are not equal !!'
    else:
        arrlen = len(predicted_arr)
    corrx = []; corry = [];   
    for k in range(arrlen):
        if(predicted_arr[k]>0 and reference_arr[k]>0):
            corrx.append(predicted_arr[k])
            corry.append(reference_arr[k])
        
    corr_coef    = pearsonr(corrx,corry)
    pearson_corr = format(corr_coef[0], '.3f')
    return pearson_corr

def rmse(predicted_arr, reference_arr):
    ### check if lengths are equal or not ###
    if(len(predicted_arr)!=len(reference_arr)):
        return 'Lengths of both arrays are not equal !!'
    else:
        arrlen = len(predicted_arr)
    diff_err = 0.0
    for k in range(arrlen):
        diff_err = diff_err + np.power((predicted_arr[k] - reference_arr[k]),2);
    
    diff_err = np.sqrt(diff_err/arrlen)
    diff_err = format(diff_err, '.3f')
    return diff_err

def corr(predicted_arr, reference_arr):
    ### check if lengths are equal or not ###
    if(len(predicted_arr)!=len(reference_arr)):
        return 'Lengths of both arrays are not equal !!'
    else:
        arrlen = len(predicted_arr)
    corrx = []; corry = [];   
    for k in range(arrlen):  
        corrx.append(predicted_arr[k])
        corry.append(reference_arr[k])
        
    corr_coef    = pearsonr(corrx,corry)
    pearson_corr = format(corr_coef[0], '.3f')
    return pearson_corr
    
if __name__ == '__main__':
    
    ### test the functions here ###
    reference_arr = [240, 243, 245, 248, 248.5]
    predicted_arr = [ 240.08501267, 242.77743394, 245.27510678, 247.77743394, 248.58501267]
    
    ### compute RMSE ###
    print 'RMSE: '+str(rmse(predicted_arr, reference_arr))
    
    ### compute correlation ###
    print 'CORR: '+str(corr(predicted_arr, reference_arr))