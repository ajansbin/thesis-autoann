from collections import defaultdict
import json
import tqdm

from pyquaternion import Quaternion
import numpy as np

def load_prediction(result_path: str):
    """
    Loads object predictions from file.
    :param result_path: Path to the .json result file provided by the user.
    """

    pred_boxes = defaultdict(list)

    # Load from file and check that the format is correct.
    with open(result_path) as f:
        data = json.load(f)
    assert 'results' in data, 'Error: No field `results` in result file.'

    # Deserialize results and get meta data.
    for sample_token, boxes in tqdm.tqdm(data['results'].items(), leave=True):
         for box in boxes:
            q = Quaternion(np.array(box['rotation']))
            box['rotation'] = convert_to_sine_cosine(q)

         pred_boxes[sample_token].append(boxes)

    meta = data['meta']

    return pred_boxes, meta

def convert_to_sine_cosine(quaternion):
    rot_sine = np.sin(quaternion.yaw_pitch_roll[0])
    rot_cosine = np.cos(quaternion.yaw_pitch_roll[0])
    return [rot_sine, rot_cosine]

def convert_to_quaternion(sine_cosine):
    q = Quaternion(axis=[0, 0, 1], radians=np.arctan2(sine_cosine[0], sine_cosine[1]))
    return q
