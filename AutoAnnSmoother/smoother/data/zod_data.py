from .common.tracking_results import TrackingResults
import os
from zod import ZodSequences
from zod.constants import Lidar
from smoother.data.loading.loader import load_prediction
from smoother.data.loading.zod_loader import load_gt
from zod.data_classes.geometry import Pose

OBJECT_CLASSES_DYNAMIC = [
    "Vehicle",
    "VulnerableVehicle",
    "Pedestrian",
    "Animal",
]
OBJECT_CLASSES_STATIC = [
    "PoleObject",
    "TrafficBeacon",
    "TrafficSign",
    "TrafficSignal",
    "TrafficGuide",
    "DynamicBarrier",
]
OBJECT_CLASSES = [
    *OBJECT_CLASSES_DYNAMIC,
    *OBJECT_CLASSES_STATIC,
    "Inconclusive",
]


class ZodTrackingResults(TrackingResults):

    def __init__(self, tracking_results_path, config, version, split, anno_path="", data_path="/data/zod", zod=None):
        print("Initializing ZodData class...")
        super(ZodTrackingResults, self).__init__(tracking_results_path, config, version, split, data_path)
        assert os.path.exists(tracking_results_path), 'Error: The result file does not exist!'
        assert version in ['full', 'mini'] 
        self.zod = ZodSequences(data_path, version) if not zod else zod
        self.anno_path = anno_path

        self.assoc_metric = config["data"]["association_metric"]
        self.assoc_thres = config["data"]["association_thresholds"][self.assoc_metric]

        assert split in ['train', 'val']
        self.seq_tokens = self.zod.get_split(split)
        # Store the FoI-indexes
        self.foi_indexes = {}
        for seq_token in self.seq_tokens:
            self.foi_indexes[seq_token] = self._get_foi_index(seq_token)

        self.motion_comp = config["data"]["annotations"]["motion_compensate"]
        self.world_coord = config["data"]["annotations"]["world_coord"]
        print("Loading prediction and ground-truths ...")
        self.pred_boxes, self.meta = self.load_tracking_predictions(self.tracking_results_path)
        self.gt_boxes = self.load_gt_detections()
        self.gt_frames = self.map_seq_id_to_gt(self.gt_boxes)

        self.object_classes = OBJECT_CLASSES

    def load_tracking_predictions(self, tracking_results_path):
        return load_prediction(tracking_results_path)
    
    def load_gt_detections(self):
        return load_gt(self.zod, 
                       self.seq_tokens, 
                       motion_compensate=self.motion_comp, 
                       world_coord=self.world_coord, 
                       verbose=True)

    def map_seq_id_to_gt(self, gt_boxes):
        gt_frames = {}
        for seq_id in gt_boxes:
            seq = self.zod[seq_id]
            annotated_frame = os.path.basename(seq.info.get_key_lidar_frame().filepath)
            gt_frames[seq_id] = annotated_frame
        return gt_frames
    
    def get_sequence_id_from_index(self, index):
        return self.seq_tokens[index]
    
    #def get_sequence_from_id(self, id):
    #    return self.zod[id]

    def get_class_label(self, class_name):
        return self.object_classes.index(class_name)

    def get_frames_in_sequence(self, seq_token):
        seq = self.zod[seq_token]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        frame_tokens = [os.path.basename(l.to_dict()['filepath']) for l in lidar_frames]
        return frame_tokens
    
    def get_pred_boxes_from_frame(self, frame_token):
        pred_boxes = self.pred_boxes[frame_token]
        return pred_boxes[0] if len(pred_boxes) > 0 else pred_boxes
    
    def get_gt_boxes_from_frame(self, frame_token):
        seq_token = frame_token[0:6]
        return self.gt_boxes.get(seq_token, [])
    
    #def get_first_frame_in_sequence(self, seq):
    #    raise NotImplementedError

    def get_number_of_sequences(self):
        return len(self.zod)
    
    def get_length_of_sequence(self, seq):
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        return len(lidar_frames)
    
    def get_timestamp_from_frame(self, frame_token):
        # a frame token looks like this 000002_romeo_2022-06-13T10:49:57.555450Z.npy. Extract time-date from name
        return
    
    def get_foi_index(self, seq_token):
        return self.foi_indexes[seq_token]
    
    def _get_foi_index(self, seq_token):
        seq = self.zod[seq_token]
        frames = self.get_frames_in_sequence(seq_token)

        lidar_frame_index = None
        annotated_frame = os.path.basename(seq.info.get_key_lidar_frame().filepath)
        for i, frame in enumerate(frames):
            if frame==annotated_frame:
                lidar_frame_index = i
        return lidar_frame_index
    
    def get_lidar_data_in_frame(self, frame_token, frame_index, lidar=True):
        seq_token = frame_token[0:6]
        seq = self.zod[seq_token]
        
        lidar_frame = seq.info.lidar_frames[Lidar.VELODYNE][frame_index]
        lidar_data = lidar_frame.read()
        if lidar:
            return lidar_data
        
        # Transform to world-coordinates
        calib = seq.calibration
        ex = calib.get_extrinsics(Lidar.VELODYNE)
        lidar_data.transform(ex)

        timestamp = lidar_frame.time.timestamp()
        #core_timestamp = lidar_data.core_timestamp
        core_ego_pose = Pose(seq.ego_motion.get_poses(timestamp))
        lidar_data.transform(core_ego_pose)

        return lidar_data
    
    def get_points_in_frame(self, frame_token, frame_index):
        points = self.get_lidar_data_in_frame(frame_token, frame_index, lidar=False).points
        return points

