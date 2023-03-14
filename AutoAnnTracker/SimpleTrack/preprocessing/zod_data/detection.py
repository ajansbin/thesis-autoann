import os, argparse, numpy as np, json
from tqdm import tqdm
from zod import ZodSequences
from zod.constants import Lidar

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_folder', type=str, default='/datasets/zod/zodv2')
parser.add_argument('--data_folder', type=str, default='../../../datasets/zod/')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--file_path', type=str, default='val.json')
parser.add_argument('--velo', action='store_true', default=False)
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
parser.add_argument('--version', type=str, default='mini', choices=['mini', 'full'])

args = parser.parse_args()


def get_sample_tokens(data_folder):
    '''result = dict()

    zod_seq = ZodSequences(args.raw_data_folder, args.version)
    split = zod_seq.get_split(args.split)
    
    for _, scene_name in enumerate(zod_seq.get_all_ids()):
        if scene_name not in split:
            continue
        seq = zod_seq[scene_name]
        lidar_frames = seq.info.get_lidar_frames(lidar=Lidar.VELODYNE)
        result[scene_name] = [l.to_dict()['filepath'] for l in lidar_frames]

    return result
    '''
    token_folder = os.path.join(data_folder, 'token_info')
    file_names = sorted(os.listdir(token_folder))
    result = dict()
    for i, file_name in enumerate(file_names):
        file_path = os.path.join(token_folder, file_name)
        scene_name = file_name.split('.')[0]
        tokens = json.load(open(file_path, 'r'))

        result[scene_name] = tokens

    return result


def sample_result2bbox_array(sample):
    trans, size, rot, score = \
        sample['translation'], sample['size'],sample['rotation'], sample['detection_score']
    return trans + size + rot + [score]


def main(det_name, file_path, detection_folder, data_folder):
    # dealing with the paths
    detection_folder = os.path.join(detection_folder, det_name)
    output_folder = os.path.join(detection_folder, 'dets')
    os.makedirs(output_folder, exist_ok=True)
    
    # load the detection file
    print('LOADING RAW FILE')
    f = open(file_path, 'r')
    det_data = json.load(f)['results']
    f.close()

    # prepare the scene names and all the related tokens
    tokens = get_sample_tokens(data_folder)
    scene_names = sorted(list(tokens.keys()))
    bboxes, inst_types, velos = dict(), dict(), dict()
    for scene_name in scene_names:
        frame_num = len(tokens[scene_name])
        bboxes[scene_name], inst_types[scene_name] = \
            [[] for i in range(frame_num)], [[] for i in range(frame_num)]
        if args.velo:
            velos[scene_name] = [[] for i in range(frame_num)]

    # enumerate through all the frames
    sample_keys = list(det_data.keys())
    print('PROCESSING...')
    pbar = tqdm(total=len(sample_keys))
    for sample_index, sample_key in enumerate(sample_keys):
        # find the corresponding scene and frame index
        scene_name, frame_index = None, None
        for scene_name in scene_names:
            if sample_key in tokens[scene_name]:
                frame_index = tokens[scene_name].index(sample_key)
                break
        
        # extract the bboxes and types
        sample_results = det_data[sample_key]
        for sample in sample_results:
            bbox, inst_type = sample_result2bbox_array(sample), sample['detection_name']
            inst_velo = sample['velocity']
            bboxes[scene_name][frame_index] += [bbox]
            inst_types[scene_name][frame_index] += [inst_type]

            if args.velo:
                velos[scene_name][frame_index] += [inst_velo]
        pbar.update(1)
    pbar.close()

    # save the results
    print('SAVING...')
    pbar = tqdm(total=len(scene_names))
    for scene_name in scene_names:
        if args.velo:
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name], velos=velos[scene_name])
        else:
            np.savez_compressed(os.path.join(output_folder, '{:}.npz'.format(scene_name)),
                bboxes=bboxes[scene_name], types=inst_types[scene_name])
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    detection_folder = os.path.join(args.data_folder, 'detection')
    os.makedirs(detection_folder, exist_ok=True)

    main(args.det_name, args.file_path, detection_folder, args.data_folder)