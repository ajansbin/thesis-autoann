import tqdm
from nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.splits import create_splits_scenes
from collections import defaultdict


def load_gt(nusc: NuScenes, eval_split: str, box_type: str, verbose: bool = False) -> dict:
    """
    Loads object predictions from file.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_type: Type of box to load, e.g. detection or tracking.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes as dictionary.
    """

    # Init.
    if box_type == 'detection':
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.format(eval_split, nusc.version))

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError('Error: Requested split {} which this function cannot map to the correct NuScenes version.'
                         .format(eval_split))
    
    # Only keep samples from this split.
    splits = create_splits_scenes()

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    gt_boxes = defaultdict(list)

    if eval_split == 'test':
        for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):
            gt_boxes[sample_token] = []
        return gt_boxes
    
    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation', sample_annotation_token)
            if box_type == 'detection':

                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name(sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # Get attribute_name.
                attr_tokens = sample_annotation['attribute_tokens']
                attr_count = len(attr_tokens)
                if attr_count == 0:
                    attribute_name = ''
                elif attr_count == 1:
                    attribute_name = attribute_map[attr_tokens[0]]
                else:
                    raise Exception('Error: GT annotations must not have more than one attribute!')
                
                this_box = {
                    "sample_token": sample_token,
                    "translation": sample_annotation["translation"],
                    "size": sample_annotation["size"],
                    "rotation": sample_annotation["rotation"],
                    "velocity":nusc.box_velocity(sample_annotation['token'])[:2],
                    "num_pts":sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    "detection_name":detection_name,
                    "detection_score":-1.0,  # GT samples do not have a score.
                    "attribute_name":attribute_name,
                    "instance_token": sample_annotation["instance_token"],
                    "token":sample_annotation["token"],
                    "name": sample_annotation["category_name"]
                            }
                            
            elif box_type == "tracking":
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                # Import locally to avoid errors when motmetrics package is not installed.
                from nuscenes.eval.tracking.utils import category_to_tracking_name
                tracking_name = category_to_tracking_name(sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                this_box = {
                    "sample_token":sample_token,
                    "translation":sample_annotation['translation'],
                    "size":sample_annotation['size'],
                    "rotation":sample_annotation['rotation'],
                    "velocity":nusc.box_velocity(sample_annotation['token'])[:2],
                    "num_pts":sample_annotation['num_lidar_pts'] + sample_annotation['num_radar_pts'],
                    "tracking_id":tracking_id,
                    "tracking_name":tracking_name,
                    "tracking_score":-1.0  # GT samples do not have a score.
                }

            else:
                raise NotImplementedError('Error: Invalid box_type %s!' % box_type)
            
            sample_boxes.append(this_box)
        gt_boxes[sample_token].append(sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(len(gt_boxes)))

    return gt_boxes