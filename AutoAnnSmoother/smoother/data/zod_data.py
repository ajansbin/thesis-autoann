from .common.sequence_data import TrackingResults
import os


class ZodTrackingResults(TrackingResults):

    def __init__(self, tracking_results_path, config, version, split, data_path="/data/zod"):
        print("Initializing NuscenesData class...")
        super(ZodTrackingResults, self).__init__(tracking_results_path, config, version, split, data_path)
        assert os.path.exists(tracking_results_path), 'Error: The result file does not exist!'

        # ToDo:Change to ZOD
        # assert version in ['v1.0-trainval', 'v1.0-mini'] 
        # if version == 'v1.0-trainval':
        #     self.train_scenes = list(splits.train)
        #     self.val_scenes = list(splits.val)
        # else: #args.version == 'v1.0-mini':
        #     self.train_scenes = list(splits.mini_train)
        #     self.val_scenes = list(splits.mini_val)

        # ToDo: Change to ZOD
        #self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True) if not nusc else nusc
        
        # ToDo: Change to ZOD
        # yields a list of all scene-tokens for the current split
        #print("Splitting data ...")
        #self.scene2split = {self.nusc.scene[i]['token']: self._get_scene2split(self.nusc.scene[i]['name']) for i in range(len(self.nusc.scene))}
        #self.split_scene_token = [scene_token for scene_token in self.scene2split if self.scene2split[scene_token] == self.split]

        print("Loading prediction and ground-truths ...")
        self.pred_boxes, self.meta = self.load_tracking_predictions(self.tracking_results_path)
        self.gt_boxes = self.load_gt_detections()

    def load_tracking_predictions(self, tracking_results_path):
        raise NotImplementedError
    
    def load_gt_detections(self):
        raise NotImplementedError
    
    def get_sequence_id_from_index(self, index):
        raise NotImplementedError
    
    def get_scene_from_id(self, id):
        raise NotImplementedError
    
    def get_frame_from_id(self, id):
        raise NotImplementedError
    
    def get_pred_boxes_from_frame(self, frame_token):
        raise NotImplementedError
    
    def get_gt_boxes_from_frame(self, frame_token):
        raise NotImplementedError
    
    def get_first_frame_in_sequence(self, seq):
        raise NotImplementedError
    
    def get_next_frame(self, prev_frame):
        raise NotImplementedError
    
    def get_number_of_sequences(self):
        raise NotImplementedError
    
    def get_length_of_sequence(self, seq):
        raise NotImplementedError

    def _get_scene2split(self, scene_name):
        if scene_name in self.train_scenes:
            return "train"
        elif scene_name in self.val_scenes:
            return "val"
        else:
            return "test"