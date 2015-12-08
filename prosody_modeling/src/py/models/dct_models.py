'''
Created on 9 Oct 2015

@author: Srikanth Ronanki
'''
import numpy as np
import scipy.fftpack as fftpack
import scipy.interpolate

#### All functions supporting DCT and IDCT operations ####
#### DCT and IDCT type-II is implemented here ############

class DCTModels(object):
    
    def __init__(self):
        pass
     
    def dct(self, arr,coef_size=0):
        ## Type-II DCT
        dcta2  = fftpack.dct(arr,norm='ortho') 
        if(coef_size==0):
            return dcta2
        arrlen = len(arr)
        if(arrlen < coef_size):
            return self.mean_fit(dcta2, coef_size)
        elif(arrlen > coef_size):
            return dcta2[0:coef_size]
        else:
            return dcta2

    def idct(self, arr,coef_size=0):
        if(coef_size==0):
            idcta2 = fftpack.idct(arr,norm='ortho')
            return idcta2
    
        arrlen = len(arr)
        if coef_size > arrlen:
            new_arr = self.interpolation_with_zeros(arr,coef_size)
        else:
            new_arr = arr[0:int(coef_size)]
        
        idcta2 = fftpack.idct(new_arr,norm='ortho')
        return idcta2

    def mean_fit(self, arr,coef_size):
        xlen = len(arr)
        x = [i for i in range(xlen)]
        y = arr
        y_interp = self.mean_interpolation(x,y)
        new_arr = [y[i] for i in range(xlen-1)]
        xtralen= coef_size-xlen
        for i in range(xtralen):
            value = xlen-2+1/(float(xtralen)+1)*(i+1)
            new_arr.append(y_interp(value))
        new_arr.append(y[xlen-1])    
        return new_arr 

    def mean_interpolation(self, x,y):
        y_interp = scipy.interpolate.interp1d(x, y)
        return y_interp 
    
    def interpolation_with_zeros(self, arr,coef_size):
        arrlen = len(arr)
        xtralen = int(coef_size) - arrlen
        lt=[]
        for x in range(xtralen):
            lt.append(x-x)
        arr=np.append(arr,lt)  
        return arr

if __name__=="__main__":
    
    #### To test the functions with different lengths ###
    
    f0_contour_a = [240, 243, 245, 248, 248.5]
    f0_contour_b = [240, 243, 245, 248, 248.5, 249, 249.2, 249.4, 250, 251, 251, 252.5, 254, 252, 250.4, 248.6, 247, 245.1, 245.0]
    
    ### set/modify parameters here ###
    coef_size   = 7
    f0_contour  = f0_contour_a
    
    ### compute length of F0 contour ###
    contour_len = len(f0_contour)
    
    ### check the transformations here ###
    print 'Original F0 contour:'
    print f0_contour
    
    models = DCTModels()
    print 'DCT-parametrized contour:'
    dct_arr = models.dct(f0_contour,coef_size)
    print dct_arr
    
    print 'Reconstructed(IDCT) F0 contour:'
    recon_arr = models.idct(dct_arr, contour_len)
    print recon_arr
    
