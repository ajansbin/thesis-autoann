import os, json
import tqdm

from zod import ZodSequences
from zod.constants import Lidar, AnnotationProject
from zod.anno.object import AnnotatedObject

from smoother.data.common.utils import convert_to_sine_cosine

VALID_OBJECTS = ["Vehicle","VulnerableVehicle","Pedestrian"]

def load_gt(zod, scene_tokens, verbose=False):
    seq_gt = {}
    for seq_id in tqdm.tqdm(scene_tokens, leave=verbose):        
        seq = zod[seq_id]
        annotations = seq.get_annotation(AnnotationProject.OBJECT_DETECTION)
        
        frame_anns = []
        for ann_obj in annotations:
            if ann_obj.box3d:
                if ann_obj.should_ignore_object(require_3d=True) or ann_obj.name not in VALID_OBJECTS:
                    continue

                translation = ann_obj.box3d.center
                size = ann_obj.box3d.size
                rotation = ann_obj.box3d.orientation
                            

                this_box = {
                    "sample_token": seq_id,
                    "translation": translation,
                    "size": size,
                    "rotation": rotation,
                    "velocity": [0.0,0.0],
                    #"num_pts":sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    "detection_name":ann_obj.name,
                    "detection_score":-1.0,  # GT samples do not have a score.
                    "attribute_name":ann_obj.object_type,
                    "instance_token": ann_obj.uuid,
                    "name": ann_obj.name
                }
            
                frame_anns.append(this_box)
        if len(frame_anns)>0:
            seq_gt[seq_id] = frame_anns
    return seq_gt
