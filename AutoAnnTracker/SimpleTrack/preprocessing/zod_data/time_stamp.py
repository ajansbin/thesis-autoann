
import os, numpy as np, zod, argparse, json
from zod import ZodSequences
from zod.constants import Lidar
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='/datasets/zod/zodv2')
parser.add_argument('--data_folder', type=str, default='../../../datasets/zod/')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--version', type=str, default='mini', choices=['mini', 'full'])
args = parser.parse_args()

def main(zod_seq, split, ts_folder):
    pbar = tqdm(total=len(split))
    for scene_index, scene_name in enumerate(zod_seq.get_all_ids()):
        if scene_name not in split:
            continue

        seq = zod_seq[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        time_stamps = [l.time.timestamp() for l in lidar_frames]
        #time_stamps = [l.to_dict()['time'] for l in lidar_frames]

        f = open(os.path.join(ts_folder, '{:}.json'.format(scene_name)), 'w')
        json.dump(time_stamps, f)
        f.close()
        pbar.update(1)
    pbar.close()
    return

if __name__ == '__main__':
    print('time stamp')
    ts_folder = os.path.join(args.data_folder, 'ts_info')
    os.makedirs(ts_folder, exist_ok=True)
    
    zod_seq = ZodSequences(args.raw_data_folder, args.version)
    split = zod_seq.get_split(args.split)

    main(zod_seq, split, ts_folder)