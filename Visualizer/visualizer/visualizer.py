from smoother.data.zod_data import ZodTrackingResults
from smoother.data.common.tracking_data import WindowTrackingData
from smoother.data.common.transformations import ToTensor, Normalize, CenterOffset, YawOffset
from smoother.data.common.utils import convert_to_quaternion
from smoother.io.config_utils import load_config
from zod.constants import Camera, Lidar, Anonymization, AnnotationProject, EGO, CoordinateFrame
from matplotlib import pyplot as plt
from zod.data_classes.box import Box3D
from zod.data_classes.sensor import LidarData
from zod.visualization.lidar_on_image import visualize_lidar_on_image
from zod.visualization.object_visualization import overlay_object_3d_box_on_image
from smoother.models.pc_track_net import PCTrackNet, PCNet, TrackNet
import torch
from PIL import Image
import os
import copy


class VisualizeResults():

    def __init__(self, conf_path, version, split, result_path, anno_path, data_path, remove_non_foi_tracks=False, remove_non_gt_tracks=False):
        self.conf_path = conf_path
        self.version = version
        self.split = split
        self.result_path = result_path
        self.anno_path = anno_path
        self.data_path = data_path
        #self.model_path = model_path

        self.det_color = (0,0,255)      # Blue
        self.ref_color = (255,255,0)    # Yellow
        self.gt_color = (255,0,0)       # Red
        self.line_thickness = 3

        self.conf = load_config(self.conf_path)

        print(self.data_path)
        self.result_data = ZodTrackingResults(self.result_path, self.conf, self.version, self.split, self.anno_path, self.data_path)
        self.transformations = self._add_transformations(self.conf["data"]["transformations"])
        self.window_size = self.conf["data"]["window_size"]
        start_ind = int(-(self.window_size-1)/2)
        end_ind = int((self.window_size-1)/2)
        self.track_data = WindowTrackingData(self.result_data, self.window_size, self.transformations, remove_non_foi_tracks=remove_non_foi_tracks, remove_non_gt_tracks=remove_non_gt_tracks)

        self.trained_model = None

    def load_model(self, model_path, new_conf_path=None):
        if new_conf_path:
            self.conf = load_config(new_conf_path)
        self.trained_model = self._get_trained_model(model_path, self.conf)
        print(f"Succefully loaded model {os.path.basename(model_path)}")

    def plot_track_index(self, track_index, frame_track_index, show_lidar=True, show_det=True, show_ref=True, show_gt=True, score_thresh=0.0):
        track, seq, track_camera_index, track_box = self._extract_track_and_sequence_info(track_index, frame_track_index, self.track_data)
        
        print("Showing track", track)

        frames =list(seq.info.get_camera_lidar_map(Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE))
        camera_frame, lidar_frame = frames[track_camera_index]
        image = camera_frame.read()

        if show_lidar:
            masked_lidar = self._get_masked_lidar(track_box, lidar_frame, seq)
            image = self._add_lidar_to_image(image, masked_lidar, seq)

        if show_det:
            track_box3d = self._create_track_box(track_box, seq)
            image = self._add_box_to_image(image, track_box3d, seq, color=self.det_color, line_thickness=self.line_thickness)

        if show_ref:
            ref_box, score = self._get_refined_box(self.trained_model, self.track_data, track_index, frame_track_index, seq, score_thresh)
            if ref_box:
                image = self._add_box_to_image(image, ref_box, seq, color=self.ref_color, line_thickness=self.line_thickness)
            else:
                print(f"Refinement score is {score} is lower than score threshold {score_thresh}.")

        if show_gt:
            if track.has_gt:
                gt_box = self._get_gt_box(track, seq)
                image = self._add_box_to_image(image, gt_box, seq, color=self.gt_color, line_thickness=self.line_thickness)
            else:
                print("Track does not have gt, skipping gt-box")


        self._plot_image(image)

    def generate_sequence_gif(self, seq_id, output_dir, gif_name="sequence_tracks.gif",step=2, duration=400, show_lidar=True, show_det=True, show_ref=True, show_gt=True, score_thresh=0.0):
        seq = self.result_data.zod[seq_id]
        camera_lidar_map = self._get_camera_lidar_index_map(seq)

        tracks_same_seq = self._get_tracks_in_seq(seq_id)
        print(f"Found {len(tracks_same_seq)} number of tracks in sequence")

        frames = list(seq.info.get_camera_lidar_map(Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE))

        images = []
        output_size = (1081, 608)  # Output image size (width, height)

        for lidar_index in range(0, camera_lidar_map[-1], step):
            camera_index = camera_lidar_map.index(lidar_index)
            camera_frame, lidar_frame = frames[camera_index]

            image = camera_frame.read()

            for track_index, track_same_seq in tracks_same_seq:
                if lidar_index < track_same_seq.starting_frame_index or lidar_index >= track_same_seq.starting_frame_index + len(track_same_seq.boxes):
                    continue

                #print(lidar_index, track_same_seq.starting_frame_index)
                frame_track_index = lidar_index - track_same_seq.starting_frame_index
                track_box = track_same_seq.boxes[frame_track_index]

                if show_lidar:
                    masked_lidar = self._get_masked_lidar(track_box, lidar_frame, seq)
                    if masked_lidar.points.shape[0] == 1 or masked_lidar.points.shape[0] > 50000:
                        continue
                    image = self._add_lidar_to_image(image, masked_lidar, seq)

                if show_det:
                    track_box3d = self._create_track_box(track_box, seq)
                    image = self._add_box_to_image(image, track_box3d, seq, color=self.det_color, line_thickness=self.line_thickness)

                if show_ref:
                    ref_box, score = self._get_refined_box(self.trained_model, self.track_data, track_index, frame_track_index, seq, score_thresh)
                    if ref_box:
                        image = self._add_box_to_image(image, ref_box, seq, color=self.ref_color, line_thickness=self.line_thickness)

                if lidar_index == track_same_seq.foi_index and show_gt:
                    if track_same_seq.has_gt:
                        gt_box = self._get_gt_box(track_same_seq, seq)
                        image = self._add_box_to_image(image, gt_box, seq, color=self.gt_color, line_thickness=self.line_thickness)
                    else:
                        print("Track does not have gt, skipping gt-box")

            image = Image.fromarray(image)
            image = image.resize(output_size, Image.ANTIALIAS)
            images.append(image)

        # Save as GIF
        gif_path = os.path.join(output_dir, gif_name)
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
        print(f"Generated sequence gif at {gif_path}")

    def _get_tracks_in_seq(self, seq_id):
        tracks_same_seq = []
        for i in range(len(self.track_data)):
            t = self.track_data.get(i)
            if t.sequence_id == seq_id:
                tracks_same_seq.append((i, t))
        return tracks_same_seq

    
    def _extract_track_and_sequence_info(self, track_index, frame_track_index, track_data):
        track = track_data.get(track_index)
        seq = self.result_data.zod[track.sequence_id]
        camera_lidar_map = self._get_camera_lidar_index_map(seq)

        track_box_index = track.foi_index - track.starting_frame_index if frame_track_index=='foi' else frame_track_index
        track_box = track.boxes[track_box_index]

        track_lidar_index = track.starting_frame_index + track_box_index
        track_camera_index = camera_lidar_map.index(track_lidar_index)

        return track, seq, track_camera_index, track_box

    def _get_camera_lidar_index_map(self, seq):
        frames = seq.info.get_camera_lidar_map(Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE)
        camera_lidar_map = []
        lidar_i = 0
        _, lidar_frame = next(frames)
        lidar_name = lidar_frame.filepath
        camera_lidar_map.append(lidar_i)
        for _, lidar_frame in frames:
            if  lidar_frame.filepath != lidar_name:
                lidar_i += 1
                lidar_name = lidar_frame.filepath
            camera_lidar_map.append(lidar_i) 

        return camera_lidar_map
    
    def _create_track_box(self, track_box, seq):
        box = Box3D(track_box.center, track_box.size, convert_to_quaternion(track_box.rotation), Lidar.VELODYNE)
        box.convert_to(Camera.FRONT, seq.calibration)
        return box

    def _get_gt_box(self, track, seq):
        # function code
        gt_box_dict = track.gt_box
        gt_translation = gt_box_dict["translation"]
        gt_size = gt_box_dict["size"]
        gt_rotation = convert_to_quaternion(gt_box_dict["rotation"])

        gt_box = Box3D(gt_translation, gt_size, gt_rotation, Lidar.VELODYNE)
        gt_box.convert_to(Camera.FRONT, seq.calibration)

        return gt_box
    
    def _get_refined_box(self, model, track_data, track_index,frame_track_index, seq, score_thresh):
        track_obj = track_data.get(track_index)
        old_foi_index = copy.copy(track_data.get(track_index).foi_index)

        frame_track_index = track.foi_index - track.starting_frame_index if frame_track_index=='foi' else frame_track_index

        track_data.get(track_index).foi_index = frame_track_index + track_obj.starting_frame_index
        track, point, gt = track_data[track_index]
        print(track.unsqueeze(0).shape, point.unsqueeze(0).shape)
        center_out, size_out, rot_out, score_out = model.forward(track.unsqueeze(0), point.unsqueeze(0))
        print(center_out.shape, size_out.shape, rot_out.shape, score_out.shape)

        center_out = center_out.squeeze(0)
        size_out = size_out.squeeze(0)
        rot_out = rot_out.squeeze(0)
        score_out = score_out.squeeze(0)

        mid_wind = track.shape[0] // 2 + 1
        c_hat = track[mid_wind, 0:3] + center_out[mid_wind]
        s_hat = track[mid_wind, 3:6] + size_out
        r_hat = track[mid_wind, 6:8] + rot_out[mid_wind]

        score = score_out[mid_wind]

        if score < score_thresh:
            return (None, score)

        model_out = torch.cat((c_hat, s_hat, r_hat, score), dim=-1).squeeze().detach()
        #unnormalize
        for transformation in reversed(self.transformations):
            if type(transformation) == CenterOffset:
                transformation.set_offset(track_data.get(track_index).center_offset)
                transformation.set_start_and_end_index(0, -1)
            if type(transformation) == Normalize:
                transformation.set_start_and_end_index(0, -1)
            model_out = transformation.untransform(model_out)

        # create refined box and add to image
        ref_box = Box3D(model_out[0:3].numpy(), model_out[3:6].numpy(), convert_to_quaternion(model_out[6:8].numpy()), Lidar.VELODYNE)

        ref_box.convert_to(Camera.FRONT, seq.calibration)

        track_data.get(track_index).foi_index = old_foi_index

        return (ref_box, score)
    
    def _get_masked_lidar(self, track_box, lidar_frame, seq):
        core_lidar = self.result_data.get_lidar_data_in_frame(track_box.frame_token, track_box.frame_index)
        core_lidar = lidar_frame.read()
        masked_points = track_box.get_points_in_bbox(core_lidar.points)

        masked_lidar = LidarData(masked_points, core_lidar.timestamps, core_lidar.intensity, core_lidar.diode_idx, core_lidar.core_timestamp)
        masked_lidar.transform(seq.calibration.get_extrinsics(EGO))

        return masked_lidar

    def _plot_image(self, image):
        plt.rcParams["figure.figsize"] = [20, 10]
        plt.axis("off")
        plt.imshow(image)
        plt.show()

    def _add_lidar_to_image(self, image, masked_lidar, seq):
        return visualize_lidar_on_image(
            masked_lidar,
            seq.calibration,
            image,
        )
    
    def _add_box_to_image(self, image, box3d, seq, color=(0, 255, 0), line_thickness=2):
        return overlay_object_3d_box_on_image(image, box3d, seq.calibration, color, line_thickness)

    def _add_transformations(self, transformations_dict):
        transformations = [ToTensor()]

        if transformations_dict["normalize"]["normalize"]:
            center_mean = transformations_dict["normalize"]["center"]["mean"]
            center_stdev = transformations_dict["normalize"]["center"]["stdev"]
            size_mean = transformations_dict["normalize"]["size"]["mean"]
            size_stdev = transformations_dict["normalize"]["size"]["stdev"]
            rotation_mean = transformations_dict["normalize"]["rotation"]["mean"]
            rotation_stdev = transformations_dict["normalize"]["rotation"]["stdev"]


            means = center_mean + size_mean + rotation_mean
            stdev = center_stdev + size_stdev + rotation_stdev
            transformations.append(Normalize(means,stdev))
        if transformations_dict["center_offset"]:
            transformations.append(CenterOffset())
        if transformations_dict["yaw_offset"]:
            transformations.append(YawOffset())

        return transformations
    
    def _get_trained_model(self, model_path, conf):
        self.use_pc = conf["model"]["pc"]["use_pc"]
        self.use_track = conf["model"]["track"]["use_track"]

        pc_encoder = conf["model"]["pc"]["pc_encoder"]
        pc_out_size = conf["model"]["pc"]["pc_out_size"]
        track_encoder = conf["model"]["track"]["track_encoder"]
        track_out_size = conf["model"]["track"]["track_out_size"]
        decoder_name = conf["model"]["decoder"]["name"]
        dec_out_size = conf["model"]["decoder"]["dec_out_size"]

        model_cls = self._get_model_cls(self.use_pc, self.use_track)

        model = model_cls(track_encoder=track_encoder, 
                                    pc_encoder=pc_encoder, 
                                    decoder=decoder_name, 
                                    pc_feat_dim=4, 
                                    track_feat_dim=9, 
                                    pc_out=pc_out_size, 
                                    track_out=track_out_size, 
                                    dec_out=dec_out_size)


        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _get_model_cls(self, use_pc, use_track):
        if use_pc and use_track:
            return PCTrackNet
        if use_pc:
            return PCNet
        if use_track:
            return TrackNet
        raise NotImplementedError("Model without point-cloud nor tracks not available")