import os, json
import tqdm

from zod import ZodSequences
from zod.constants import Lidar
from zod.anno.object import AnnotatedObject

from pyquaternion import Quaternion
from smoother.data.loading.loader import convert_to_sine_cosine

VALID_OBJECTS = ["Vehicle","VulnerableVehicle","Pedestrian"]

def load_gt(data_path, scene_tokens, verbose=False):
    sequence_path = os.path.join(data_path, "sequences")
    sequences = [seq for seq in os.listdir(sequence_path) if seq in scene_tokens]

    seq_gt = {}
    for sequence in tqdm.tqdm(sequences, leave=verbose):
        frame_path = os.path.join(sequence_path, sequence, "annotations/object_detection/")
        
        if not os.path.exists(frame_path):
            continue
        
        frame_json = os.listdir(frame_path)[0]

        frame_token  = frame_json
        seq_token = frame_token[0:6]
        json_path = os.path.join(frame_path,frame_json)
        with open(json_path) as f:
            anns = json.load(f)

        frame_anns = []
        ignore_counter = 0
        for i, ann in enumerate(anns):
            ann_obj = AnnotatedObject.from_dict(ann)
            if ann_obj.should_ignore_object(require_3d=True) or ann_obj.name not in VALID_OBJECTS:
                ignore_counter += 1
                continue


            translation = ann_obj.box3d.center
            size = ann_obj.box3d.size
            rotation = ann_obj.box3d.orientation
                        

            this_box = {
                "sample_token": frame_token,
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
        
        #print(f"Ignored {ignore_counter} annotations for sequence {sequence}, length {len(anns)}")
        if len(frame_anns)>0:
            seq_gt[seq_token] = frame_anns
    return seq_gt