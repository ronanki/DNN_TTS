#import matplotlib
#matplotlib.use('macosx') 
from matplotlib import pyplot as pylab
import sys

import numpy as np
import scipy.signal
import scipy.interpolate



#import pylab


_SMOOTH = 1
_DETREND = 2
_TREND = 3
def smooth(params, win, mode = _SMOOTH):
    
    """
    gaussian smoothing
    """
    if win >= len(params)-1:
        win = len(params)-1
    if win % 2 != 0:
        win+=1
    
    s = np.r_[params[win-1:0:-1],params,params[-1:-win:-1]]
    w = np.hamming(win)
    
        
    y = np.convolve(w/w.sum(),s,mode='valid')

    if mode == _DETREND:
        
        yy = y[(win/2-1):-(win/2)]
        return params-yy

    elif mode == _TREND:
        return y[(win/2-1):-(win/2)]
    else: 
        return y[(int(round(win/2))-1):-(int(round(win/2)))]

def detrend(params, win):
    return smooth(params, win, _DETREND)

def trend(params, win):
    return smooth(params, win, _TREND)

def remove_spikes(params, thresh):
    despiked = np.array(params)
    st_params = semitones(params)
    smoothed = smooth(st_params,10)
    # be more carefule with upward spikes, they may be emphatic accents
    for i in range(0, len(smoothed)):
        if (st_params[i] - smoothed[i]) >  thresh:
            despiked[i] = 0
        
        if (smoothed[i] - st_params[i]) > thresh:
            despiked[i] = 0
            
    return interpolate_zeros(despiked, 'linear')




def fix_octave_jumps2(f0, win = 50, thresh=7):
    fixed = np.array(f0)
    st_params = semitones(f0)
    for i in range(2, len(f0)-2):
        slice = st_params[max(i-win, 0):min(i+win, len(f0)-1)]
        median = np.median(slice[slice > 0])

        if (median - st_params[i]) > thresh:
            fixed[i] = 0
            
        if (median -st_params[i]) < - thresh:
            fixed[i] = 0
    return interpolate_zeros(fixed, 'linear')

def fill_gaps(params, max_speed = 3, max_gap_len =50):
    """
    based on max derivative
    # max_speed: st / frame
    """
    filled = np.array(params)
    derivate = smooth(np.diff(semitones(params), 1), 10)
    i = 0
    while i < len(derivate)-max_gap_len:
        if derivate[i] < -max_speed:

            for j in range(i+max_gap_len, i, -1):
                if derivate[j] > max_speed:
                    
                    filled[i-1:j+1] = 0
                    i = j+1
                    break

        i = i +1
    return interpolate_zeros(filled, 'linear')

def cut_below_mean(params, win=31):
    smoothed = smooth(params, win) #
    #smoothed = scipy.signal.medfilt(params, win)
    for i in range(0, len(params)):
        if params[i] > smoothed[i]:

            smoothed[i]=params[i]
        else:
            smoothed[i] = 0
            pass
        #smoothed[i] = params[i]

    return interpolate_zeros(smoothed, 'linear')
    #return smoothed

def replace_with_voiced_median(params, f0, win):
    #pylab.plot(params)
    replaced = np.array(params)
    replaced[0:win] = np.mean(params)
    replaced[-win:] = np.mean(params)
  
    for i in range(win, len(params)-win):
        if f0[i] == 0:
            slice= params[i-win:i+win]
            f0_slice= f0[i-win:i+win]
            try:
                replaced[i] = np.median(slice[f0_slice > 0])
            except:
                replaced[i] = np.median(params)
            if np.isnan(replaced[i]):
                replaced[i] = np.median(params)
    replaced[0:win] = np.mean(replaced)
    replaced[-win:] = np.mean(replaced)
  
    #replaced = cut_below_mean(replaced,30)
  
    #pylab.plot(replaced)
    return replaced

def semitones(params, base = 40):
    semitones = np.array(params)
    for i in range(0, len(params)):
        if params[i] > 0:
            semitones[i] =  (12 * np.log2(params[i] / base) / np.log2(2))
    return semitones

#def hz(params, base=40)
#    for i in range(0, len(params)):


def pad_with_noise(params, f0):
    first = 0
    last = -1
    mean = np.mean(params[f0>0])
    min = np.min(params[f0>0])
    var = np.std(params[f0>0])
    padded = np.array(params)
    for i in range(0, len(f0)):
         if f0[i] > 0:
             first = i
             break
    for i in range(len(params)-1, 0, -1):
        if f0[i] > 0:
            last = i
            break
        
    for i in range(0,first):
        padded[i] = mean - (np.random.rand() -0.5)*var
    for i in range(last, len(params)):
        padded[i] = mean - (np.random.rand() -0.5)*var
    return padded

def constrain_interpolation(params, large_gap = 60):

    constrained = np.array(params)
    voiced = params[params > 0]
    constrained[0] = np.mean(voiced)
    constrained[-1] = np.mean(voiced)
    gap = 0
    prev_voiced = 0.0
    for i in range(0, len(params)):
        if constrained[i] == 0 and i < len(params-2):
            gap = gap + 1
        else:
            if gap >= large_gap:
                slice = params[max(i-gap-50, 0):min(i+50, len(params)-2)]
                constrained[i-gap+20:i-20] = np.min(slice[slice>0])
            gap = 0
            prev_voiced = constrained[i]
    return constrained

def replace_unvoiced(f0, params):
    replaced = np.array(f0)
    for i in range(0, len(params)):
        if f0[i] == 0:
            replaced[i] = params[i]
    return replaced

def interpolate_zeros(params, method='spline', min_val = 0):

        #min_val = np.nanmin(params)
        voiced = np.array(params, float)        
	for i in range(0, len(voiced)):
		if voiced[i] <= min_val:
			voiced[i] = np.nan
        #voiced[0] =  scipy.stats.nanmean(voiced) #[1:int(len(voiced))]) #1 #min_val
        #voiced[-1] = np.nanmin(voiced[len(voiced)*0.75:len(voiced)-1]) #1 #min_val
	if np.isnan(voiced[-1]):
		voiced[-1] = np.nanmin(voiced)
	if np.isnan(voiced[0]):
            voiced[0] = scipy.stats.nanmean(voiced)

        not_nan = np.logical_not(np.isnan(voiced))

        indices = np.arange(len(voiced))
        if method == 'spline':
            interp = scipy.interpolate.UnivariateSpline(indices[not_nan],voiced[not_nan], k=2) #, s=0.5)
            # return voiced parts intact
            smoothed = interp(indices)
            for i in range(0, len(smoothed)):
                if params[i] > min_val:
                    smoothed[i] = params[i]
            return smoothed
        else:
            interp = scipy.interpolate.interp1d(indices[not_nan], voiced[not_nan], method)
        return interp(indices)
    



def process(f0):

    #pylab.plot(f0)
    # try yo fix octave_jumps
    params = fix_octave_jumps2(f0, 60, 6)
    # set long unvoiced to median
    params = constrain_interpolation(params, 30)
    
    
    
    
    #interpolate
    interpolated = interpolate_zeros(scipy.signal.medfilt(params, 11), 'linear') # spline often better but goes wild on long gaps
    interpolated = replace_unvoiced(f0, interpolated)
    
    # remove spikes caused by interpolation
    interpolated = remove_spikes(interpolated,1.5) 
    #fill gaps
    interpolated = fill_gaps(interpolated,1.0, 30)
    interpolated = fix_octave_jumps2(interpolated, 60, 6)
    #pylab.plot(interpolated)
    #raw_input()
    return interpolated
        
# test
"""
f0= np.loadtxt(sys.argv[1])
pylab.plot(f0)
pylab.plot(process_f0(f0), label="processed")
pylab.legend()
raw_input()
"""
