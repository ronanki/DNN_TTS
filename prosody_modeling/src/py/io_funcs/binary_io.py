

import numpy

class   BinaryIOCollection(object):

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size / dimension))]
        features = features.reshape((-1, dimension))
            
        return  features

    def load_float_file(self, file_name):
        fid_feat = open(file_name,'r')
        feat_arr = [float(x.strip()) for x in fid_feat.readlines()]
        fid_feat.close()
        return feat_arr
    
    def load_file_list(self, file_name):
        fid_feat = open(file_name,'r')
        list_arr = [x.strip() for x in fid_feat.readlines()]
        fid_feat.close()
        return list_arr
    
    def array_to_binary_file(self, data, output_file_name):
        data = numpy.array(data, 'float32')
               
        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()    

    def load_binary_file_frame(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        frame_number = features.size / dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))
            
        return  features, frame_number

