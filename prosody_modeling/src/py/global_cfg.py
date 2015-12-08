import os

class globalConfig(object):
    
    def __init__(self):
        self.work_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/Hybrid_prosody_model'
        self.dnn_dir = '/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/dnn_tts_blzpilot/'
        self.vowelListPath = os.path.join(self.work_dir, 'Data/phoneset/all_phone_list.txt') 
        