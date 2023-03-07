import os, json
import mmcv

from zod import ZodSequences
from zod.constants import Lidar

from collections import defaultdict


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
            if not mmcv.is_filepath(lidar_path):
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

def load_prediction(result_path: str) \
        -> Tuple[Dict, Dict]:
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    :param max_boxes_per_sample: Maximim number of boxes allowed per sample.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The deserialized results and meta data.
    """

    pred_boxes = defaultdict(list)

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file. Please note that the result format changed.' \
                              'See https://www.nuscenes.org/object-detection for more information.'

    # Deserialize results and get meta data.

    for sample_token, boxes in data['results'].items():
         pred_boxes[sample_token].append(boxes)

    meta = data['meta']

    return pred_boxes, meta