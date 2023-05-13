import os, json
import tqdm
import mmcv
from zod import ZodSequences
from zod.constants import Lidar, AnnotationProject
from zod.anno.object import AnnotatedObject
from zod.data_classes.box import Box3D
from zod.constants import Lidar, EGO, AnnotationProject, Anonymization
from zod.data_classes.geometry import Pose

VALID_OBJECTS = ["Vehicle", "VulnerableVehicle", "Pedestrian"]


def load_gt(zod, scene_tokens, motion_compensate=True, world_coord=True, verbose=False):
    seq_gt = {}

    for seq_id in tqdm.tqdm(scene_tokens, position=0, leave=True):
        seq = zod[seq_id]
        annotations = seq.get_annotation(AnnotationProject.OBJECT_DETECTION)
        frame_anns = []
        for ann_obj in annotations:
            if (
                ann_obj.should_ignore_object(require_3d=True)
                or ann_obj.name not in VALID_OBJECTS
            ):
                continue

            translation = ann_obj.box3d.center
            size = ann_obj.box3d.size
            rotation = ann_obj.box3d.orientation

            box = Box3D(translation, size, rotation, Lidar.VELODYNE)

            if motion_compensate:
                cam_frame = seq.info.get_key_camera_frame(Anonymization.BLUR)
                cam_timestamp = cam_frame.time.timestamp()
                pose_camera = Pose(seq.ego_motion.get_poses(cam_timestamp))

                lid_frame = seq.info.get_key_lidar_frame()
                lid_timestamp = lid_frame.time.timestamp()
                pose_lidar = Pose(seq.ego_motion.get_poses(lid_timestamp))

                box.convert_to(EGO, seq.calibration)
                box._transform(pose_camera, EGO)

                box._transform_inv(pose_lidar, EGO)
                box.convert_to(Lidar.VELODYNE, seq.calibration)

            if world_coord:
                box.convert_to(EGO, seq.calibration)
                box._transform(pose_lidar, EGO)

            box = {
                "sample_token": seq_id,
                "translation": list(box.center),
                "size": list(box.size),
                "rotation": list(box.orientation.elements),
                "velocity": [0.0, 0.0],
                "detection_name": ann_obj.name,
                "detection_score": -1.0,  # GT samples do not have a score.
                "attribute_name": ann_obj.object_type,
                "instance_token": ann_obj.uuid,
                "name": ann_obj.name,
            }

            frame_anns.append(box)
        if len(frame_anns) > 0:
            seq_gt[seq_id] = frame_anns
    return seq_gt
