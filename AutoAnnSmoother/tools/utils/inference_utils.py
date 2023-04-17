from smoother.data.common.sequence_data import SequenceData
import torch
import os
import numpy as np
from smoother.data.common.dataclasses import Tracklet
from smoother.data.common.transformations import ToTensor

class SequenceInferer():

    def __init__(self, seq_data:SequenceData, foi_index:int):
        self.seq_data = seq_data
        self.foi_index = foi_index

        self.use_pc = self.seq_data.use_pc
        self.data_conf = self.seq_data.data_conf
        self.pc_offset = self.seq_data.pc_offset

        self.data_samples = list(self._get_data_samples())
        self.max_track_length = self.seq_data.max_track_length
        self.transformations = self.seq_data.transformations
        self.center_offset_index = self.seq_data.center_offset_index
        self.yaw_offset_index = self.seq_data.yaw_offset_index
        self.normalize_index = self.seq_data.normalize_index

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, index):
        track = self.data_samples[index]

        track_data = self._create_track_data(track)
        track_data = self._apply_transformations(track, track_data)
        if self.use_pc:
            track_points = self._get_and_pad_track_points(track)
        else:
            track_points = torch.tensor([])
        return track_data, track_points

    def _create_track_data(self, track):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        track_data = [self._get_track_data_entry(i, track, track_start_index, track_end_index) for i in range(self.max_track_length)]

        return track_data

    def _get_track_data_entry(self, i, track, track_start_index, track_end_index):
        if i < track_start_index or i >= track_end_index:
            pad_data = [0] * 8 + [i - self.foi_index]
            return pad_data
        else:
            box = track[i - track_start_index]
            temporal_encoding = [box.frame_index - self.foi_index]
            return box.center + box.size + box.rotation + temporal_encoding

    def _get_and_pad_track_points(self, track):
        track_points = self._get_point_clouds(track)
        if self.pc_offset:
            track_points = self._get_offset_pc(track, track_points)

        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)
        if track_end_index == 181:
            track_end_index = 180

        # Initialize a tensor with the required dimensions
        full_track_points = torch.zeros((self.max_track_length, track_points.shape[1], track_points.shape[2] + 1), dtype=torch.float32)

        # Fill in the track_points
        full_track_points[track_start_index:track_end_index, :, :-1] = track_points[0:track_end_index-track_start_index]

        # Fill in the temporal encoding for all points
        temporal_encoding = torch.arange(180) - 89

        # Unsqueezing the temporal encoding to match the dimensions of full_track_points
        temporal_encoding = temporal_encoding.unsqueeze(1).unsqueeze(2)
        # Adding the temporal encoding as a new column in the last dimension of full_track_points
        full_track_points[:, :, -1:] = full_track_points[:, :, -1:] + temporal_encoding

        return full_track_points.float()

    
    def _get_point_clouds(self, track):
        if "pc_path" in self.data_conf:
            root_pc_path = self.data_conf["pc_path"]
        else:
            #root_pc_path = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed/preprocessed_full_train'
            root_pc_path = '/staging/agp/masterthesis/2023autoann/storage/smoothing/autoannsmoothing/preprocessed/preprocessed_mini_train'

        pc_name = f"point_clouds_{track.sequence_id}_{track.tracking_id}.npy"
        pc_path = os.path.join(root_pc_path, pc_name)
        pc = torch.from_numpy(np.load(pc_path)).float()

        if pc.shape[0] == 181:
            pc = pc[:180,:]
        return pc

    def _get_offset_pc(self, track: Tracklet, track_points: np.ndarray):
        foi_box = track.boxes[self.foi_index-track.starting_frame_index]
        foi_center = np.array(foi_box.center)
        offset_points = track_points - foi_center
        return offset_points

    def _apply_transformations(self, track, track_data):
        track_start_index = track.starting_frame_index
        track_end_index = track_start_index + len(track)

        for i, transformation in enumerate(self.transformations):
            if i == self.center_offset_index:
                foi_data = track_data[self.foi_index]
                transformation.set_offset(foi_data)
                track.set_center_offset(transformation.offset)
                transformation.set_start_and_end_index(track_start_index, track_end_index)
            if i == self.yaw_offset_index:
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


        

    