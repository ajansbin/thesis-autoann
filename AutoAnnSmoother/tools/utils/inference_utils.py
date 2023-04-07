from smoother.data.common.sequence_data import SequenceData
import torch

class SequenceInferer():

    def __init__(self, seq_data:SequenceData, foi_index:int):
        self.seq_data = seq_data
        self.foi_index = foi_index

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = self.seq_data.max_track_length
        self.transformations = self.seq_data.transformations
        self.center_offset_index = self.seq_data.center_offset_index
        self.normalize_index = self.seq_data.normalize_index

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, index):
        track = self.data_samples[index]
        
        track_start_index = track.starting_frame_index
        track_end_index = track.starting_frame_index + len(track)
        track_data = []
        for i in range(self.max_track_length):
            if i < track_start_index or i >= track_end_index:
                pad_data = [0]*8 + [i-self.foi_index]
                track_data.append(pad_data)
            else:
                box = track[i-track_start_index]
                temporal_encoding = [box.frame_index - self.foi_index]
                track_data.append(box.center + box.size + box.rotation + temporal_encoding)

        for i, transformation in enumerate(self.transformations):
            if i == self.center_offset_index:
                foi_data = track_data[self.foi_index]
                transformation.set_offset(foi_data)
                track.set_center_offset(transformation.offset)
            elif i == self.yaw_offset_index:
                foi_data = track_data[self.foi_index]
                transformation.set_offset(foi_data)
                track.set_yaw_offset(transformation.offset)
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            elif i == self.normalize_index:
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            track_data = transformation.transform(track_data)

        return track_data
    
    def _get_data_samples(self):
        for track in self.seq_data:
            track_start_index = track.starting_frame_index
            track_end_index = track_start_index + len(track)

            if self.foi_index < track_start_index or self.foi_index > track_end_index:
                continue

            yield track

    def get(self, track_index):
        return self.data_samples[track_index]


class WindowInferer():

    def __init__(self, seq_data:SequenceData, foi_index:int, window_start, window_end, sequence_inferer=None):
        self.seq_data = seq_data
        self.foi_index = foi_index
        self.window_start = window_start
        self.window_end = window_end

        self.sequence_inferer = SequenceInferer(self.seq_data, self.foi_index) if not sequence_inferer else sequence_inferer

    def __len__(self):
        return len(self.sequence_inferer)

    def __getitem__(self, index):
        track_data = self.sequence_inferer[index]
        start_index = self.foi_index + self.window_start # 
        end_index = self.foi_index + self.window_end + 1# 182

        device = track_data.device

        if start_index < 0:
            zero_pad_length = abs(start_index)
            zero_pad = torch.zeros((zero_pad_length,track_data.shape[-1])).to(device)
            track_data = torch.cat((zero_pad, track_data), dim=0)
            end_index += zero_pad_length #4
            start_index = 0 # 0
        if end_index > self.sequence_inferer.max_track_length:
            pad_length = end_index-self.sequence_inferer.max_track_length + 1
            zero_pad = torch.zeros((abs(pad_length),track_data.shape[-1])).to(device)
            track_data = torch.cat((track_data, zero_pad), dim=0)
            end_index = -1

        wind_track_data = track_data[start_index:end_index]

        offset_tensor = torch.arange(self.window_start, self.window_end + 1).float().view(-1,1).to(device)
        wind_track_data[:, -1] = offset_tensor.squeeze()

        return wind_track_data
    
    def get(self, track_index):
        return self.sequence_inferer.get(track_index)
    

class SlidingWindowInferer():

    def __init__(self, seq_data:SequenceData, foi_index:int, window_size:int):
        self.seq_data = seq_data
        self.foi_index = foi_index
        self.window_size = window_size

        self.sequence_inferer = SequenceInferer(self.seq_data, self.foi_index)

        self.data_samples = list(self._get_data_samples())

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, index):
        track_data = self.data_samples[index]

        return track_data

    def _get_data_samples(self):

        for i in range(self.window_size):
            start_ind = i - self.window_size + 1
            end_ind = start_ind + self.window_size - 1

            wind_inf = WindowInferer(self.seq_data, self.foi_index, start_ind, end_ind, self.sequence_inferer)
            for track in wind_inf:                
                yield track

    def get(self, sliding_track_index): 
        track_index = sliding_track_index % len(self.sequence_inferer)  
        return self.sequence_inferer.get(track_index)


        

    