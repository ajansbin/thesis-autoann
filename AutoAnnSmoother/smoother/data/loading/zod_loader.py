import os, json
import tqdm

from zod import ZodSequences
from zod.constants import Lidar
from zod.anno.object import AnnotatedObject

def get_available_scenes(zod: ZodSequences, split):

    available_scenes = []
    print('total scene num: {}'.format(len(zod.get_all_ids())))

    for scene_index, scene_name in enumerate(zod.get_all_ids()):
        if scene_name not in split:
            continue
        seq = zod[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        
        scene_not_exist = False
        for frame in lidar_frames:
            lidar_path = frame.to_dict()['filepath']
            if os.getcwd() in lidar_path:
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
            if not os.path.exists(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        #available_scenes.append(seq)
        available_scenes.append(scene_name)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def load_gt(ann_path, scene_tokens, verbose=False):
    #ann_path = "/staging/agp/masterthesis/2023autoann/storage/zod_annotations/annotations/"
    sequences = [seq for seq in os.listdir(ann_path) if seq in scene_tokens]

    seq_gt = {}
    for sequence in tqdm.tqdm(sequences, leave=verbose):
        frame_path = os.path.join(ann_path, sequence, "annotations/dynamic_objects/")
        frame_json = os.listdir(frame_path)[0]
        frame_token  = frame_json.replace(".json", ".npy")
        json_path = os.path.join(frame_path,frame_json)
        with open(json_path) as f:
            anns = json.load(f)

        frame_anns = []
        for ann in anns:
            ann_obj = AnnotatedObject.from_dict(ann)
            if ann_obj.should_ignore_object(require_3d=True):
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
        
        if len(frame_anns)>0:
            seq_gt[frame_token] = frame_anns
    return seq_gt

