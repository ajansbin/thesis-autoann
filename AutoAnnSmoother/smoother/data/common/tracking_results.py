# This class will be inherited for all different datatypes e.g. nuscenes and zod. 
class TrackingResults():

    def __init__(self, tracking_results_path, config, version="v1.0-trainval", split="val", data_path="/data/nuscenes"):
        self.tracking_results_path = tracking_results_path
        self.config = config
        self.version = version
        self.split = split
        self.data_path = data_path

        # CONFIG SPECIFIC
        self.window_size = self.config["data"]["window_size"]

        self.assoc_metric = self.config["data"]["association_metric"]
        self.gt_assoc_threshold = self.config["data"]["association_thresholds"][self.assoc_metric]

        self.remove_bottom_center = self.config["data"]["remove_bottom_center"]

        self.score_dist_temp = self.config["data"]["score_dist_temp"]

    def load_tracking_predictions(self, tracking_results_path):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def load_gt_detections(self):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_sequence_id_from_index(self, index):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    #def get_sequence_from_id(self, id):
    #    raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_frames_in_sequence(self, scene_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_pred_boxes_from_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_gt_boxes_from_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    #def get_first_frame_in_sequence(self, seq):
    #    raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_number_of_sequences(self):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")

    def get_length_of_sequence(self, seq):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")
    
    def get_timestamp_from_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")
    
    def get_foi_index(self, seq_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")
    
    def _get_points_in_frame(self, frame_token):
        raise NotImplementedError("Calling Abstract Class Method! Instead, must use child of TrackingResults.")
