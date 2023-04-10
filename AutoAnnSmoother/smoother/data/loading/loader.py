from collections import defaultdict
import json
import tqdm

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
         pred_boxes[sample_token].append(boxes)

    meta = data['meta']

    return pred_boxes, meta
