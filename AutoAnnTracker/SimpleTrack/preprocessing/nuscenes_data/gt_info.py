import os, numpy as np, nuscenes, argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from copy import deepcopy
from tqdm import tqdm
import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='../../../raw/nuscenes/data/sets/nuscenes/')
parser.add_argument('--data_folder', type=str, default='../../../datasets/nuscenes/')
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--version', type=str, default='mini', choices=['v1.0-mini', 'v1.0-test', 'v1.0-trainval'])
parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
args = parser.parse_args()


def instance_info2bbox_array(info):
    translation = info.center.tolist()
    size = info.wlh.tolist()
    rotation = info.orientation.q.tolist()
    return translation + size + rotation


def main(nusc, scene_names, root_path, gt_folder):
    pbar = tqdm(total=len(scene_names))
    for scene_index, scene_info in enumerate(nusc.scene):
        scene_name = scene_info['name']
        if scene_name not in scene_names:
            continue

        first_sample_token = scene_info['first_sample_token']
        last_sample_token = scene_info['last_sample_token']
        frame_data = nusc.get('sample', first_sample_token)
        if args.mode == '20hz':
            cur_sample_token = frame_data['data']['LIDAR_TOP']
        elif args.mode == '2hz':
            cur_sample_token = deepcopy(first_sample_token)
        
        frame_index = 0
        IDS, inst_types, bboxes = list(), list(), list()
        while True:
            frame_ids, frame_types, frame_bboxes = list(), list(), list()
            if args.mode == '2hz':
                frame_data = nusc.get('sample', cur_sample_token)
                lidar_token = frame_data['data']['LIDAR_TOP']
                instances = nusc.get_boxes(lidar_token)
                for inst in instances:
                    frame_ids.append(inst.token)
                    frame_types.append(inst.name)
                    frame_bboxes.append(instance_info2bbox_array(inst))

            elif args.mode == '20hz':
                frame_data = nusc.get('sample_data', cur_sample_token)
                lidar_data = nusc.get('sample_data', cur_sample_token)
                instances = nusc.get_boxes(lidar_data['token'])
                for inst in instances:
                    frame_ids.append(inst.token)
                    frame_types.append(inst.name)
                    frame_bboxes.append(instance_info2bbox_array(inst))
            
            IDS.append(frame_ids)
            inst_types.append(frame_types)
            bboxes.append(frame_bboxes)

            # clean up and prepare for the next
            cur_sample_token = frame_data['next']
            if cur_sample_token == '':
                break

        np.savez_compressed(os.path.join(gt_folder, '{:}.npz'.format(scene_name)), 
            ids=IDS, types=inst_types, bboxes=bboxes)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    print('gt info')
    gt_folder = os.path.join(args.data_folder, 'gt_info')
    os.makedirs(gt_folder, exist_ok=True)

    if args.version == 'v1.0-trainval':
        split = args.split
    elif args.version == 'v1.0-mini':
        split = 'mini_' + args.split
    elif args.version == 'v1.0-test':
        split = 'test'
    
    #val_scene_names = splits.create_splits_scenes()['val']
    #nusc = NuScenes(version='v1.0-trainval', dataroot=args.raw_data_folder, verbose=True)
    scene_names = splits.create_splits_scenes()[split]
    nusc = NuScenes(version=args.version, dataroot=args.raw_data_folder, verbose=True)
    main(nusc, scene_names, args.raw_data_folder, gt_folder)