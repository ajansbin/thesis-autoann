import argparse
import os
import mmcv
import tqdm
#import copy

from zod.data_classes.geometry import Pose
from zod.zod_sequences import ZodSequences
from zod.constants import Lidar, EGO
from zod.data_classes.box import Box3D


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ZOD result to world coordinates')
    parser.add_argument('--result-path', help='result file path')
    parser.add_argument('--root-path', help='data path', default='/datasets/zod/zodv2/')
    parser.add_argument('--save-path', help='save path')
    parser.add_argument('--version', default = 'mini', choices=['full', 'mini'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--convert-to', default = 'world', choices=['world', 'lidar'])
    parser.add_argument('--type', default = 'detections', choices=['detections', 'tracks'])
    args = parser.parse_args()

    return args

def main(result_path, root_path, version, split, save_path, type, convert_to):

    data = mmcv.load(result_path)
    
    zod_seq = ZodSequences(root_path, version)

    result_world_coord = {}
    result_world_coord['meta'] = data['meta']

    #result_world_coord['results'] = copy.deepcopy(data['results'])

    result_world_coord['results'] = {}
    i = 0
    for scene_name in tqdm.tqdm(zod_seq.get_split(split), leave=True):
        seq = zod_seq[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        
        for lidar_frame in lidar_frames[:-1]:
            filepath = os.path.basename(lidar_frame.filepath)
            if filepath not in list(data['results'].keys()):
                continue
            else:
                key = filepath
    
            result_world_coord['results'][key] = []

            core_timestamp = lidar_frame.time.timestamp()
            core_ego_pose = Pose(seq.ego_motion.get_poses(core_timestamp))

            for i, b in enumerate(data['results'][key]):
                if convert_to == 'world':
                    box = Box3D(b['translation'],b['size'], b['rotation'], Lidar.VELODYNE)
                    box.convert_to(EGO, seq.calibration)
                    box._transform(core_ego_pose, EGO)
                elif convert_to == 'lidar':
                    box = Box3D(b['translation'],b['size'], b['rotation'], EGO)
                    box._transform_inv(core_ego_pose, EGO)
                    box.convert_to(Lidar.VELODYNE, seq.calibration)
                
                if type == 'detections':
                    info = {
                        'sample_token': data['results'][key][i]['sample_token'],
                        'translation': list(box.center),
                        'size': list(box.size),
                        'rotation': box.orientation.elements,
                        'velocity': data['results'][key][i]['velocity'],
                        'detection_name': data['results'][key][i]['detection_name'],
                        'detection_score': data['results'][key][i]['detection_score'],
                        'attribute_name': data['results'][key][i]['attribute_name']
                    }
                elif type == 'tracks':
                    info = {
                        'sample_token': data['results'][key][i]['sample_token'],
                        'translation': list(box.center),
                        'size': list(box.size),
                        'rotation': box.orientation.elements,
                        'velocity': data['results'][key][i]['velocity'],
                        'tracking_id': data['results'][key][i]['tracking_id'],
                        'tracking_name': data['results'][key][i]['tracking_name'],
                        'tracking_score': data['results'][key][i]['tracking_score']
                    }
                
                result_world_coord['results'][key].append(info)
    mmcv.dump(result_world_coord, save_path)

if __name__ == '__main__':
    args = parse_args()

    main(args.result_path, args.root_path, args.version, args.split, args.save_path, args.type, args.convert_to)
