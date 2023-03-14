
import os, numpy as np, zod, argparse, json
from zod import ZodSequences
from zod.data_classes import sequence
from zod.data_classes.geometry import Pose
from zod.constants import Lidar
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='/datasets/zod/zodv2')
parser.add_argument('--data_folder', type=str, default='../../../datasets/zod/')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--version', type=str, default='mini', choices=['mini', 'full'])
args = parser.parse_args()

def main(zod_seq, split, ego_folder):
    pbar = tqdm(total=len(split))
    for scene_index, scene_name in enumerate(zod_seq.get_all_ids()):
        if scene_name not in split:
            continue

        seq = zod_seq[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        
        ego_data = dict()
        
        for i in range(len(seq.ego_motion.poses)):
            ego_pose = seq.ego_motion.poses[i]
            ego_pose = Pose(ego_pose)
            
            translation = ego_pose.translation 
            res = np.append(translation, ego_pose.rotation_matrix)
            ego_data[str(i)] = res

        np.savez_compressed(os.path.join(ego_folder, '{:}.npz'.format(scene_name)), **ego_data)
        pbar.update(1)
    pbar.close()
    return

if __name__ == '__main__':
    print('ego info')
    ego_folder = os.path.join(args.data_folder, 'ego_info')
    os.makedirs(ego_folder, exist_ok=True)

    
    zod_seq = ZodSequences(args.raw_data_folder, args.version)
    split = zod_seq.get_split(args.split)

    main(zod_seq, split, ego_folder)