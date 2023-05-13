from tools.utils.lidar_bev_seq import BEVBoxAnimation

from smoother.data.zod_data import ZodTrackingResults
from smoother.data.common.tracking_data import WindowTrackingData
from smoother.data.common.transformations import (
    ToTensor,
    Normalize,
    CenterOffset,
    YawOffset,
)
from smoother.io.config_utils import load_config
from zod.constants import (
    Camera,
    Lidar,
    Anonymization,
)

from smoother.models.pc_track_net import (
    PCTrackNet,
    PCNet,
    TrackNet,
    PCTrackEarlyFusionNet,
)
import torch
from PIL import Image
import os
import numpy as np
from visualizer_tools import VisualizerTools


class VisualizeResults:
    def __init__(
        self,
        conf_path,
        version,
        split,
        result_path,
        data_path,
        remove_non_foi_tracks=False,
        remove_non_gt_tracks=False,
    ):
        self.conf_path = conf_path
        self.version = version
        self.split = split
        self.result_path = result_path
        self.data_path = data_path

        self.conf = load_config(self.conf_path)
        if "pc_path" in self.conf["data"]:
            del self.conf["data"]["pc_path"]

        self.result_data = ZodTrackingResults(
            self.result_path, self.conf, self.version, self.split, self.data_path
        )

        self.transformations = self._add_transformations(
            self.conf["data"]["transformations"]
        )
        self.window_size = self.conf["data"]["window_size"]

        self.use_pc = self.conf["model"]["pc"]["use_pc"]
        self.early_fuse = self.conf["model"]["early_fuse"]

        self.track_data = WindowTrackingData(
            tracking_results=self.result_data,
            window_size=self.window_size,
            times=1,
            random_slides=False,
            use_pc=self.use_pc,
            transformations=self.transformations,
            points_transformations=[],
            remove_non_foi_tracks=remove_non_foi_tracks,
            remove_non_gt_tracks=remove_non_gt_tracks,
            seqs=None,
        )

        self.trained_model = None

        self.vis_tools = VisualizerTools(
            self.result_data, self.track_data, self.trained_model
        )

    def load_model(self, model_path, new_conf_path=None):
        if new_conf_path:
            self.conf = load_config(new_conf_path)
        self.trained_model = self._get_trained_model(model_path, self.conf)
        self.vis_tools.set_trained_model(self.trained_model)
        print(f"Succefully loaded model {os.path.basename(model_path)}")

    def plot_track_index(
        self,
        track_index,
        frame_track_index,
        show_lidar=True,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        (
            track,
            seq,
            track_camera_index,
            track_box,
        ) = self.vis_tools._extract_track_and_sequence_info(
            track_index, frame_track_index, self.track_data
        )

        print("Showing track", track)

        frames = list(
            seq.info.get_camera_lidar_map(
                Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE
            )
        )
        camera_frame, lidar_frame = frames[track_camera_index]
        image = camera_frame.read()

        if show_det:
            track_box3d = self.vis_tools._create_track_box(track_box, lidar_frame, seq)
            image = self.vis_tools._add_box_to_image(
                image,
                track_box3d,
                seq,
                color=self.det_color,
                line_thickness=self.line_thickness,
            )

        if show_lidar:
            masked_lidar = self.vis_tools._get_masked_lidar(
                track_box3d,
                track_box.frame_token,
                track_box.frame_index,
                lidar_frame,
                seq,
            )
            image = self.vis_tools._add_lidar_to_image(image, masked_lidar, seq)

        if show_ref:
            ref_box, score = self.vis_tools._get_refined_box(
                self.trained_model,
                self.track_data,
                track_index,
                frame_track_index,
                lidar_frame,
                seq,
                score_thresh,
            )
            if ref_box:
                image = self.vis_tools._add_box_to_image(
                    image,
                    ref_box,
                    seq,
                    color=self.ref_color,
                    line_thickness=self.line_thickness,
                )
            else:
                print(
                    f"Refinement score is {score} is lower than score threshold {score_thresh}."
                )

        if show_gt:
            if track.has_gt:
                gt_box = self.vis_tools._get_gt_box(track, lidar_frame, seq)
                image = self.vis_tools._add_box_to_image(
                    image,
                    gt_box,
                    seq,
                    color=self.gt_color,
                    line_thickness=self.line_thickness,
                )
            else:
                print("Track does not have gt, skipping gt-box")

        self.vis_tools._plot_image(image)

    def plot_sequence_foi(
        self,
        seq_id,
        show_lidar=True,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        seq = self.result_data.zod[seq_id]
        camera_lidar_map = self.vis_tools._get_camera_lidar_index_map(seq)

        tracks_same_seq = self.vis_tools._get_tracks_in_seq(seq_id)
        print(f"Found {len(tracks_same_seq)} number of tracks in sequence")

        frames = list(
            seq.info.get_camera_lidar_map(
                Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE
            )
        )

        # Select the frame of interest
        lidar_foi_index = self.result_data.foi_indexes[seq_id]
        camera_foi_index = camera_lidar_map.index(lidar_foi_index)
        camera_frame, lidar_frame = frames[camera_foi_index]

        image = self.vis_tools.plot_frame(
            seq,
            tracks_same_seq,
            lidar_foi_index,
            camera_frame,
            lidar_frame,
            show_lidar=show_lidar,
            show_det=show_det,
            show_ref=show_ref,
            show_gt=show_gt,
            score_thresh=score_thresh,
        )

        self.vis_tools._plot_image(image)

    def generate_sequence_gif(
        self,
        seq_id,
        output_dir,
        gif_name="sequence_tracks.gif",
        step=2,
        duration=400,
        show_lidar=True,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        seq = self.result_data.zod[seq_id]
        camera_lidar_map = self.vis_tools._get_camera_lidar_index_map(seq)

        tracks_same_seq = self.vis_tools._get_tracks_in_seq(seq_id)
        print(f"Found {len(tracks_same_seq)} number of tracks in sequence")

        frames = list(
            seq.info.get_camera_lidar_map(
                Anonymization.BLUR, Camera.FRONT, Lidar.VELODYNE
            )
        )

        images = []
        output_size = (1081, 608)  # Output image size (width, height)

        for lidar_index in range(0, camera_lidar_map[-1], step):
            camera_index = camera_lidar_map.index(lidar_index)
            camera_frame, lidar_frame = frames[camera_index]

            image = self.vis_tools.plot_frame(
                seq,
                tracks_same_seq,
                lidar_index,
                camera_frame,
                lidar_frame,
                show_lidar=show_lidar,
                show_det=show_det,
                show_ref=show_ref,
                show_gt=show_gt,
                score_thresh=score_thresh,
            )

            image = Image.fromarray(image)
            image = image.resize(output_size, Image.ANTIALIAS)
            images.append(image)

        # Save as GIF
        gif_path = os.path.join(output_dir, gif_name)
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
        )
        print(f"Generated sequence gif at {gif_path}")

    def _add_transformations(self, transformations_dict):
        transformations = [ToTensor()]

        if transformations_dict["center_offset"]:
            transformations.append(CenterOffset())
        if transformations_dict["yaw_offset"]:
            transformations.append(YawOffset())

        if transformations_dict["normalize"]["normalize"]:
            center_mean = transformations_dict["normalize"]["center"]["mean"]
            center_stdev = transformations_dict["normalize"]["center"]["stdev"]
            size_mean = transformations_dict["normalize"]["size"]["mean"]
            size_stdev = transformations_dict["normalize"]["size"]["stdev"]
            rotation_mean = transformations_dict["normalize"]["rotation"]["mean"]
            rotation_stdev = transformations_dict["normalize"]["rotation"]["stdev"]

            means = center_mean + size_mean + rotation_mean
            stdev = center_stdev + size_stdev + rotation_stdev
            transformations.append(Normalize(means, stdev))

        return transformations

    def _get_trained_model(self, model_path, conf):
        self.early_fuse = conf["model"]["early_fuse"]["early_fuse"]
        fuse_encoder = conf["model"]["early_fuse"]["fuse_encoder"]
        encoder_out_size = conf["model"]["early_fuse"]["encoder_out_size"]

        self.use_pc = conf["model"]["pc"]["use_pc"]
        pc_encoder = conf["model"]["pc"]["pc_encoder"]
        pc_out_size = conf["model"]["pc"]["pc_out_size"]

        self.use_track = conf["model"]["track"]["use_track"]
        track_encoder = conf["model"]["track"]["track_encoder"]
        track_out_size = conf["model"]["track"]["track_out_size"]

        temporal_encoder = conf["model"]["temporal_encoder"]
        dec_out_size = conf["model"]["dec_out_size"]

        # decoder_name = self.conf["model"]["decoder"]["name"]
        # dec_out_size = self.conf["model"]["decoder"]["dec_out_size"]

        if self.early_fuse:
            model = PCTrackEarlyFusionNet(
                fuse_encoder,
                encoder_out_size,
                temporal_encoder,
                dec_out_size,
                track_feat_dim=8,
                pc_feat_dim=4,
                window_size=self.window_size,
            )
        elif self.use_pc and self.use_track:
            model = PCTrackNet(
                track_encoder_name=track_encoder,
                pc_encoder_name=pc_encoder,
                temporal_encoder_name=temporal_encoder,
                track_feat_dim=8,
                pc_feat_dim=4,
                track_out=track_out_size,
                pc_out=pc_out_size,
                dec_out_size=dec_out_size,
                window_size=self.window_size,
            )

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def _get_model(self, early_fuse, use_pc, use_track):
        if early_fuse:
            return PCTrackEarlyFusionNet
        if use_pc and use_track:
            return PCTrackNet
        if use_pc:
            return PCNet
        if use_track:
            return TrackNet
        raise NotImplementedError("Model without point-cloud nor tracks not available")

    def create_bev_animation(
        self,
        seq_id,
        save_path,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        seq = self.result_data.zod[seq_id]
        bevs = []
        objects = []

        # tot_nr_frames = len(seq.info.get_lidar_frames())
        # frame_idx = int((tot_nr_frames-nr_frames)/2)
        # frames = seq.info.get_lidar_frames()[frame_idx:-frame_idx-1]

        tracks_same_seq = self.vis_tools._get_tracks_in_seq(seq_id)

        for lidar_index, lidar_frame in enumerate(seq.info.get_lidar_frames()):
            if lidar_index < 85 or lidar_index > 90:
                continue
            pcd = lidar_frame.read()
            frame_id = os.path.basename(lidar_frame.filepath)

            visualize_boxes = []

            for track_index, track_same_seq in tracks_same_seq:
                if (
                    lidar_index < track_same_seq.starting_frame_index
                    or lidar_index
                    >= track_same_seq.starting_frame_index + len(track_same_seq.boxes)
                ):
                    continue

                frame_track_index = lidar_index - track_same_seq.starting_frame_index
                track_box = track_same_seq.boxes[frame_track_index]

                if show_det:
                    track_box3d = self.vis_tools._create_track_box(
                        track_box, lidar_frame, seq
                    )
                    track_id = track_box.tracking_id
                    # visualize_boxes.append((track_box3d, 'Vehicle', track_id))
                    visualize_boxes.append((track_box3d, "Vehicle", "det"))

                if show_ref:
                    ref_box, score = self.vis_tools._get_refined_box(
                        self.trained_model,
                        self.track_data,
                        track_index,
                        frame_track_index,
                        lidar_frame,
                        seq,
                        score_thresh,
                    )
                    track_id = track_box.tracking_id
                    # visualize_boxes.append((ref_box, 'Vehicle', track_id))
                    visualize_boxes.append((ref_box, "Vehicle", "ref"))

                if show_gt:
                    lidar_key_frame = seq.info.get_key_lidar_frame()
                    key_frame_id = os.path.basename(lidar_key_frame.filepath)

                    if frame_id == key_frame_id:
                        gt_box = self.vis_tools._get_gt_box(
                            track_same_seq, lidar_frame, seq
                        )
                        visualize_boxes.append((gt_box, "Vehicle", "gt"))

                bevs.append(np.hstack((pcd.points, pcd.intensity[:, None])))
                objects.append(
                    (
                        np.array([obj[1] for obj in visualize_boxes]),
                        np.concatenate(
                            [obj[0].center[None, :] for obj in visualize_boxes], axis=0
                        ),
                        np.concatenate(
                            [obj[0].size[None, :] for obj in visualize_boxes], axis=0
                        ),
                        np.array([obj[0].orientation for obj in visualize_boxes]),
                        np.array([obj[2] for obj in visualize_boxes]),
                    )
                )
        bev = BEVBoxAnimation()
        print("Starting BEV visualizing")
        bev(bevs, objects, save_path, True)

    def plot_bev(
        self,
        seq_id,
        lidar_index,
        save_path,
        show_det=True,
        show_ref=True,
        show_gt=True,
        score_thresh=0.0,
    ):
        seq = self.result_data.zod[seq_id]
        bevs = []
        objects = []

        tracks_same_seq = self.vis_tools._get_tracks_in_seq(seq_id)

        lidar_frame = seq.info.get_lidar_frames()[lidar_index]
        pcd = lidar_frame.read()
        frame_id = os.path.basename(lidar_frame.filepath)

        visualize_boxes = []

        for track_index, track_same_seq in tracks_same_seq:
            if (
                lidar_index < track_same_seq.starting_frame_index
                or lidar_index
                >= track_same_seq.starting_frame_index + len(track_same_seq.boxes)
            ):
                continue

            frame_track_index = lidar_index - track_same_seq.starting_frame_index
            track_box = track_same_seq.boxes[frame_track_index]

            if show_det:
                track_box3d = self.vis_tools._create_track_box(
                    track_box, lidar_frame, seq
                )
                track_id = track_box.tracking_id
                # visualize_boxes.append((track_box3d, 'Vehicle', track_id))
                visualize_boxes.append((track_box3d, "Vehicle", "det"))

            if show_ref:
                ref_box, score = self.vis_tools._get_refined_box(
                    self.trained_model,
                    self.track_data,
                    track_index,
                    frame_track_index,
                    lidar_frame,
                    seq,
                    score_thresh,
                )
                track_id = track_box.tracking_id
                # visualize_boxes.append((ref_box, 'Vehicle', track_id))
                visualize_boxes.append((ref_box, "Vehicle", "ref"))

            if show_gt:
                lidar_key_frame = seq.info.get_key_lidar_frame()
                key_frame_id = os.path.basename(lidar_key_frame.filepath)

                if frame_id == key_frame_id:
                    gt_box = self.vis_tools._get_gt_box(
                        track_same_seq, lidar_frame, seq
                    )
                    visualize_boxes.append((gt_box, "Vehicle", "gt"))

            bevs.append(np.hstack((pcd.points, pcd.intensity[:, None])))
            objects.append(
                (
                    np.array([obj[1] for obj in visualize_boxes]),
                    np.concatenate(
                        [obj[0].center[None, :] for obj in visualize_boxes], axis=0
                    ),
                    np.concatenate(
                        [obj[0].size[None, :] for obj in visualize_boxes], axis=0
                    ),
                    np.array([obj[0].orientation for obj in visualize_boxes]),
                    np.array([obj[2] for obj in visualize_boxes]),
                )
            )

        bev = BEVBoxAnimation()
        print("Starting BEV visualizing")
        bev(bevs, objects, save_path, False)
