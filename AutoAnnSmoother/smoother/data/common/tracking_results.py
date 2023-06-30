# This class will be inherited for all different datatypes e.g. nuscenes and zod.
class TrackingResultsBase:
    def __init__(
        self, tracking_results_path, config, version, split, data_path="/data/zod"
    ):
        self.tracking_results_path = tracking_results_path
        self.config = config
        self.version = version
        self.split = split
        self.data_path = data_path

    def load_tracking_predictions(self, tracking_results_path):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def load_gt_detections(self):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_sequence_id_from_index(self, index):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_frames_in_sequence(self, scene_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_pred_boxes_from_frame(self, frame_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_gt_boxes_from_frame(self, frame_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_number_of_sequences(self):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_length_of_sequence(self, seq):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_timestamp_from_frame(self, frame_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def get_foi_index(self, seq_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )

    def _get_points_in_frame(self, frame_token):
        raise NotImplementedError(
            "Calling Abstract Class Method! Instead, must use child of TrackingResultsBase."
        )
