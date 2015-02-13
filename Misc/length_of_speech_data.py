import sys,os,math
import scipy.io.wavfile as sciwav
import numpy as np

if __name__ == "__main__":

    if len(sys.argv)<2:
        print 'Usage: python length_of_speech_data.py <input_wav_dir> <optional: file_list>\nIf given, file list shouldn\'t contain any extension\n'
        sys.exit(0)

    wav_dir = sys.argv[1]

    ##### opening list of files ######

    if len(sys.argv)==2:
        os.system('ls '+wav_dir+'/ > temp')
        ip1 = open('temp','r')
    else:
        file_list = sys.argv[2]
        ip1 = open(file_list,'r')

    total_length = 0.0
    count=0

    ##### processing each file #####

    for i in ip1.readlines():
        if len(sys.argv)==2:
            fname = i.strip()
        else:
            fname = i.strip()+'.wav'

        wav_file = wav_dir+'/'+fname
        count = count+1

        ##### read wav file #####

        [rate,data] = sciwav.read(wav_file) 
        flen = (float(len(data))/float(rate))
        total_length = total_length + flen 
        print fname+' '+str(round(flen,2))+' sec.'

        ### uncomment below line while you are running for first time
        #break; ### breaks after processing one file - to check errors

    #### print data size information #####
    
    print 'Total number of files: '+str(count)
    print 'Sampling frequency: '+str(rate)+' Hz'
    print 'Average length of each file: '+str(round((total_length/float(count)),2))+' sec.'
    n_mins = round(total_length/60,2)
    n_hrs = int(math.floor(total_length/3600))
    n_rem_mins = n_mins - n_hrs*60
    print 'Total length of database: '+str(int(n_mins))+' min. ('+str(n_hrs)+' hr. '+str(int(n_rem_mins))+' min.)'
    ip1.close()

    if len(sys.argv)==2:
        os.system('rm temp')
