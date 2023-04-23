import torch
from smoother.data.common.dataclasses import Tracklet

class Transformation():

    def transform(self):
        raise NotImplementedError('Calling method for abstract class Transformations')

    def untransform(self, x):
        return x
    
class ToTensor(Transformation):

    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def transform(self, x) -> torch.tensor:
        return torch.tensor(x,dtype=torch.float32).to(self.device)

class Normalize(Transformation):

    def __init__(self, means:list, stds:list, device=torch.device('cpu')):
        self.means = torch.tensor(means, dtype=torch.float32).to(device)
        self.stds = torch.tensor(stds, dtype=torch.float32).to(device)

        self.start_index = 0
        self.end_index = -1

    def transform(self,x:torch.tensor) -> torch.tensor:
        if x.ndim > 1: #temporal encoding and score not normalized
            x[self.start_index:self.end_index] = (x[self.start_index:self.end_index]-self.means)/self.stds
        else:
            x[0:-1] = (x[0:-1]-self.means)/self.stds
        return x

    def untransform(self,x:torch.tensor) -> torch.tensor:
        if x.ndim > 1: #temporal encoding and score not normalized
            x[self.start_index:self.end_index] = (x[self.start_index:self.end_index]*self.stds)+self.means
        else:
            x[0:-1] = (x[0:-1]*self.stds + self.means)
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
        if torch.is_tensor(x):
            self.offset = x[0:3].clone().detach()
        else:
            self.offset = torch.tensor(x[0:3], dtype=torch.float32)

    def set_start_and_end_index(self,start_index,end_index):
        self.start_index = start_index
        self.end_index = end_index

class YawOffset(Transformation):

    def __init__(self):
        self.offset = None

        self.start_index = 0
        self.end_index = -1

    def transform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index:self.end_index,6:8] = x[self.start_index:self.end_index,6:8] - self.offset        
        else:
            x[6:8] = x[6:8] - self.offset
        return x
    
    def untransform(self,x:torch.tensor) -> torch.tensor:
        if len(x.shape) > 1:
            x[self.start_index:self.end_index,6:8] = x[self.start_index:self.end_index,6:8] + self.offset        
        else:
            x[6:8] = x[6:8] + self.offset
        return x
    
    def set_offset(self, x:list):
        if torch.is_tensor(x):
            self.offset = x[6:8].clone().detach()
        else:
            self.offset = torch.tensor(x[6:8], dtype=torch.float32)

    def set_start_and_end_index(self,start_index,end_index):
        self.start_index = start_index
        self.end_index = end_index
