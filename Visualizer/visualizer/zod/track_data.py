from zod import ZodSequences
from zod.data_classes.sequence import ZodSequence
from zod.data_classes.box import Box3D
from typing import Literal
from zod.constants import Camera, Lidar
from visualizer.tools.utils import lidar_bev_seq
from visualizer.tools import visualizer_plt
from pyquaternion import Quaternion

import ffmpeg
import argparse

from pyquaternion import Quaternion
import numpy as np
import os, json


class ZodTrackSequence:
    def __init__(self, data_path, track_path, save_dir, name, seq_id, is_tracks, version='full', zod=None):
        
        self.is_tracks = is_tracks
        self.seq_id = seq_id
        self.file_name = name if name != '' else seq_id

        self.save_path = self._get_save_path(save_dir)
        self.tracks = self._load_track_res(track_path)
        
        self.zod = ZodSequences(data_path, version) if not zod else zod
        self.frames = self._get_frames_of_sequence()
        self.frameid_to_boxes = self._create_track_and_frame_dict(self.seq_id, self.tracks)


    def _get_frames_of_sequence(self):
        seq = self.zod[self.seq_id]
        frames = []
        for lidar_frame in seq.info.get_lidar_frames()[25:-25]:
            frames.append(os.path.basename(lidar_frame.filepath))
        return frames
    
    def _create_track_and_frame_dict(self, seq_id, detections):
        print('Create track and frame to Box3D mapping')
        #trackid_to_boxes={}
        frameid_to_boxes = {}

        #if detections used with frames [0:-1]
        if self.is_tracks:
            frames = self._get_frames_of_sequence()[0:-1]
        else:
            frames = self._get_frames_of_sequence()[0:-1]
        print(len(frames))
        for i, frame in enumerate(frames):
            if frame not in frameid_to_boxes:
                frameid_to_boxes[frame] = []
            
            dets=detections['results'][frame]

            #for box in dets:
            #    if box['tracking_id'] not in trackid_to_boxes:
            #        trackid_to_boxes[box['tracking_id']] = []
                
        for i, frame in enumerate(frames):
            dets=detections['results'][frame]

            for box in dets:
                center = np.array(box['translation'])
                size = np.array(box['size'])
                orientation = Quaternion(box['rotation'])
                #coord_frame = Camera.FRONT
                coord_frame = Lidar.VELODYNE
                box3d = Box3D(center, size, orientation, coord_frame)
                if self.is_tracks:
                    track_id = box['tracking_id']
                    name = box['tracking_name']
                else:
                    track_id = box['detection_name']
                    name = box['detection_name']
                #only visualizing vehicle
                if name != 'Vehicle' or box['tracking_score'] < 0.2:
                    continue
                frameid_to_boxes[frame].append((box3d, name, track_id))
                #trackid_to_boxes[box['tracking_id']].append((box3d, box['tracking_name']))
        #return trackid_to_boxes, frameid_to_boxes
        return frameid_to_boxes
    
    def _get_save_path(self, save_dir):
        save_path = os.path.join(save_dir, self.file_name +'.gif')
        return save_path
    
    def _load_track_res(self, track_path):
        print('Loading results')
        with open(track_path, 'r') as f:
            tracks = json.load(f)
        return tracks 
    
    def create_lidar_animation(self, nr_frames):
        print('Creating BEV animation for', self.seq_id)
        seq = self.zod[self.seq_id]
        bevs = []
        objects = []
        for i, lidar_frame in enumerate(seq.info.get_lidar_frames()[25:-25]):
            pcd = lidar_frame.read()
            
            #if to use aggregated frames
            #if i-2 < 0 or i+2>len(seq.info.get_lidar_frames()):
            #    continue
            #pcd = seq.get_aggregated_lidar(i-2, i+2)
            #print('frameid_to_boxes', self.frameid_to_boxes)
            frame_id = os.path.basename(lidar_frame.filepath)
            visualize_boxes = self.frameid_to_boxes[frame_id]
            #print('visualize_boxes', visualize_boxes)
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
                np.array([obj[2] for obj in visualize_boxes]),
            ))
            if i == nr_frames:
                break
        bev = lidar_bev_seq.BEVBoxAnimation()
        bev(bevs, objects, self.save_path, self.is_tracks)

'''
    def bev_detections(self, nr_frames):
        visualizer = Visualizer2D(name='', figsize=(12, 12))

        seq = self.zod[self.seq_id]

        for i, lidar_frame in enumerate(seq.info.get_lidar_frames()[0:nr_frames]):
            frame_id = os.path.basename(lidar_frame.filepath)
            #print('self.frameid_to_boxes', self.frameid_to_boxes)
            visualize_boxes = self.frameid_to_boxes[frame_id]
            
            for obj in visualize_boxes:
                box3d = Box3D(obj[0].center, obj[0].size, Quaternion(obj[0].orientation), Lidar)
                visualizer.handler_box(box3d, message='', color='red')

            #print('visualize_boxes', visualize_boxes)

        save_folder = '/home/s0001668/workspace/storage/tracking/visualized'
        visualizer.save(os.path.join(save_folder, 'detections_globinv.png'))

    def bev_detections_animations(self, nr_frames):
        #visualizer = Visualizer2D(name='', figsize=(12, 12))
        save_folder = '/home/s0001668/workspace/storage/tracking/visualized/detections'

        seq = self.zod[self.seq_id]

        for i, lidar_frame in enumerate(seq.info.get_lidar_frames()[0:nr_frames]):
            visualizer = visualizer_plt.Visualizer2D(name='', figsize=(12, 12))
            
            frame_id = os.path.basename(lidar_frame.filepath)
            #print('self.frameid_to_boxes', self.frameid_to_boxes)
            visualize_boxes = self.frameid_to_boxes[frame_id]
            
            for obj in visualize_boxes:
                box3d = Box3D(obj[0].center, obj[0].size, Quaternion(obj[0].orientation), Lidar)
                visualizer.handler_box(box3d, message='', color='red')

            #print('visualize_boxes', visualize_boxes)

            print('save_folder', save_folder)
            visualizer.save(os.path.join(save_folder, str(i)+'.png'))
            visualizer.close()
  


        framerate = 1
        file_path = save_folder +'/*.png'
        out_name = save_folder + '/movie.mp4'
        imgs = ffmpeg.input(file_path,  pattern_type='glob', framerate=framerate)
        out = ffmpeg.output(imgs, out_name)

        out.run()
        print("... Video rendering completed!")
'''