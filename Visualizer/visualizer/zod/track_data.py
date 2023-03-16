from zod import ZodSequences
from zod.data_classes.sequence import ZodSequence
from zod.data_classes.box import Box3D
from zod.constants import Camera
from visualizer.tools.utils import lidar_bev_seq


from pyquaternion import Quaternion
import numpy as np
import os, json


class ZodTrackSequence:
    def __init__(self, data_path, track_path, save_dir, seq_id, version='full', zod=None):
        
        self.seq_id = seq_id

        self.save_path = self._get_save_path(save_dir)
        self.tracks = self._load_track_res(track_path)
        
        self.zod = ZodSequences(data_path, version) if not zod else zod
        self.frames = self._get_frames_of_sequence()
        self.trackid_to_boxes, self.frameid_to_boxes = self._create_track_and_frame_dict(self.seq_id, self.tracks)


    def _get_frames_of_sequence(self):
        seq = self.zod[self.seq_id]
        frames = []
        for lidar_frame in seq.info.get_lidar_frames():
            frames.append(os.path.basename(lidar_frame.filepath))
        return frames
    
    def _create_track_and_frame_dict(self, seq_id, detections):
        print('Create track and frame to Box3D mapping')
        trackid_to_boxes={}
        frameid_to_boxes = {}

        frames = self._get_frames_of_sequence()

        for i, frame in enumerate(frames):
            if frame not in frameid_to_boxes:
                frameid_to_boxes[frame] = []
            dets=detections['results'][frame]

            for box in dets:
                if box['tracking_id'] not in trackid_to_boxes:
                    trackid_to_boxes[box['tracking_id']] = []
                
        for i, frame in enumerate(frames):
            dets=detections['results'][frame]

            for box in dets:
                center = np.array(box['translation'])
                size = np.array(box['size'])
                orientation = Quaternion(box['rotation'])
                coord_frame = Camera.FRONT
                box3d = Box3D(center, size, orientation, coord_frame)
                frameid_to_boxes[frame].append((box3d, box['tracking_name']))
                trackid_to_boxes[box['tracking_id']].append((box3d, box['tracking_name']))
        return trackid_to_boxes, frameid_to_boxes
    
    def _get_save_path(self, save_dir):
        save_path = os.path.join(save_dir, self.seq_id +'.gif')
        return save_path
    
    def _load_track_res(self, track_path):
        print('Loading tracking results')
        with open(track_path, 'r') as f:
            tracks = json.load(f)
        return tracks 
    
    def create_bev_animation(self, nr_frames):
        print('Creating BEV animation for', self.seq_id)
        seq = self.zod[self.seq_id]
        bevs = []
        objects = []
        for i, lidar_frame in enumerate(seq.info.get_lidar_frames()):
            pcd = lidar_frame.read()
            
            #if to use aggregated frames
            #if i-2 < 0 or i+2>len(seq.info.get_lidar_frames()):
            #    continue
            #pcd = seq.get_aggregated_lidar(i-2, i+2)
            
            frame_id = os.path.basename(lidar_frame.filepath)
            visualize_boxes = self.frameid_to_boxes[frame_id]
            
            bevs.append(np.hstack((pcd.points, pcd.intensity[:, None])))
            objects.append((
                np.array([obj[1] for obj in visualize_boxes]),
                np.concatenate(
                    [obj[0].center[None, :] for obj in visualize_boxes], axis=0
                ),
                np.concatenate(
                    [obj[0].size[None, :] for obj in visualize_boxes], axis=0
                ),
                np.array([obj[0].orientation for obj in visualize_boxes]),
            ))
            if i == nr_frames:
                break
        bev = lidar_bev_seq.BEVBoxAnimation()
        print('save_path', self.save_path)
        bev(bevs, objects, self.save_path)