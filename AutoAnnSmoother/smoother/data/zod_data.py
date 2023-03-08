from .common.sequence_data import TrackingResults
import os
from zod import ZodSequences
from zod.constants import Lidar
from smoother.data.loading.loader import load_prediction


class ZodTrackingResults(TrackingResults):

    def __init__(self, tracking_results_path, config, version, split, data_path="/data/zod", zod=None):
        print("Initializing NuscenesData class...")
        super(ZodTrackingResults, self).__init__(tracking_results_path, config, version, split, data_path)
        assert os.path.exists(tracking_results_path), 'Error: The result file does not exist!'

        self.zod = ZodSequences(data_path, version) if not zod else zod
        self.frame_idx = 0
        # ToDo:Change to ZOD
        assert version in ['full', 'mini'] 
        self.train_scenes = self.zod.get_split('train')
        self.val_scenes = self.zod.get_split('val')      

        self.scene_tokens = list(self.zod.get_all_ids())  
        
        #self.split_scene_token = set(get_available_scenes(zod, train_scenes))
        #val_scenes = set(get_available_scenes(zod, val_scenes))
        
        # ToDo: Change to ZOD
        # yields a list of all scene-tokens for the current split
        #print("Splitting data ...")
        #self.scene2split = {self.nusc.scene[i]['token']: self._get_scene2split(self.nusc.scene[i]['name']) for i in range(len(self.nusc.scene))}
        #self.split_scene_token = [scene_token for scene_token in self.scene2split if self.scene2split[scene_token] == self.split]

        print("Loading prediction and ground-truths ...")
        self.pred_boxes, self.meta = self.load_tracking_predictions(self.tracking_results_path)
        self.gt_boxes = self.load_gt_detections()

    def load_tracking_predictions(self, tracking_results_path):
        return load_prediction(tracking_results_path)
    
    def load_gt_detections(self):
        return [] #no annotations yet
    
    def get_sequence_id_from_index(self, index):
        return self.scene_tokens[index]
    
    #def get_sequence_from_id(self, id):
    #    return self.zod[id]

    def get_frames_in_sequence(self, scene_token):
        seq = self.zod[scene_token]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        frame_tokens = [os.path.basename(l.to_dict()['filepath']) for l in lidar_frames]
        return frame_tokens
    
    def get_pred_boxes_from_frame(self, frame_token):
        return self.pred_boxes[frame_token][0]
    
    def get_gt_boxes_from_frame(self, frame_token):
        return [] #no annotations yet
    
    #def get_first_frame_in_sequence(self, seq):
    #    raise NotImplementedError

    def get_number_of_sequences(self):
        return len(self.zod)
    
    def get_length_of_sequence(self, seq):
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        return len(lidar_frames)
