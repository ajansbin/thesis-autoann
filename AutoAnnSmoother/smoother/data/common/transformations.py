import torch
from smoother.data.common.dataclasses import Tracklet

class Transformation():

    def transform(x, x_other=None):
        raise NotImplementedError('Calling method for abstract class Transformations')
    
    def set_offset(self, x):
        pass

    def untransform(self, x):
        return x
    
class ToTensor(Transformation):

    def __init__(self):
        pass

    def transform(self, x) -> torch.tensor:
        return torch.tensor(x,dtype=torch.float32)

class Normalize(Transformation):

    def __init__(self, means:list, stds:list):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds = torch.tensor(stds, dtype=torch.float32)

        self.start_index = 0
        self.end_index = -1

    def transform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
             x[self.start_index:self.end_index] = (x[self.start_index:self.end_index,:]-self.means)/self.stds
        else:
            x = (x-self.means)/self.stds
        return x

    def untransform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
             x[self.start_index:self.end_index] = (x[self.start_index:self.end_index,:]*self.stds)+self.means
        else:
            x = (x*self.stds + self.means)
        return x
    
    def set_start_and_end_index(self,start_index,end_index):
        self.start_index = start_index
        self.end_index = end_index


class CenterOffset(Transformation):

    def __init__(self):
        self.offset = None

        self.start_index = 0
        self.end_index = -1

    def transform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index:self.end_index,0:3] = x[self.start_index:self.end_index,0:3] - self.offset        
        else:
            x[0:3] = x[0:3] - self.offset
        return x
    
    def untransform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index:self.end_index,0:3] = x[self.start_index:self.end_index,0:3] + self.offset        
        else:
            x[0:3] = x[0:3] + self.offset
        return x
    
    def set_offset(self, x:list):
        self.offset = torch.tensor(x[0:3], dtype=torch.float32)

    def set_start_and_end_index(self,start_index,end_index):
        self.start_index = start_index
        self.end_index = end_index
