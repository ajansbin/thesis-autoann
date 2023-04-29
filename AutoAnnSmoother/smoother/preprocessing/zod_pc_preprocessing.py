import os
import numpy as np
import tqdm
from smoother.data.common.dataclasses import TrackingBox
from smoother.data.zod_data import ZodTrackingResults
from smoother.io.config_utils import load_config
from zod import ZodSequences
from collections import defaultdict
import multiprocessing
from smoother.data.common.utils import convert_to_yaw
import copy


def preprocess(data_path, version, split, save_dir, tracking_result_path):
    assert os.path.exists(
        tracking_result_path
    ), "Error: The result file does not exist!"
    assert version in ["full", "mini"]
    zod = ZodSequences(data_path, version)

    assert split in ["train", "val"]
    seq_tokens = zod.get_split(split)

    conf_path = "/AutoAnnSmoother/configs/training_config.yaml"
    conf = load_config(conf_path)
    tracking_results = ZodTrackingResults(
        tracking_result_path, conf, version, split, data_path=data_path
    )

    preprocess_and_save_point_clouds(tracking_results, save_dir)


def process_sequence(sequence_token, tracking_results, N_max, save_dir):
    track_point_clouds = defaultdict(list)
    sequence_frames = tracking_results.get_frames_in_sequence(sequence_token)[:-1]
    remove_bottom_center = tracking_results.config["data"]["remove_bottom_center"]

    for frame_index, frame_token in enumerate(sequence_frames):
        frame_pred_boxes = tracking_results.get_pred_boxes_from_frame(frame_token)
        if frame_pred_boxes == []:
            continue

        points = tracking_results.get_points_in_frame(frame_token, frame_index)

        for b in frame_pred_boxes:
            box = copy.deepcopy(b)
            box["is_foi"] = False
            box["frame_index"] = frame_index
            box["frame_token"] = frame_token
            if remove_bottom_center:
                box["translation"][-1] = box["translation"][-1] + box["size"][-1] / 2
            box["rotation"] = convert_to_yaw(box["rotation"])
            tracking_box = TrackingBox.from_dict(box)

            points_masked = tracking_box.get_points_in_bbox(points)

            track_key = (sequence_token, tracking_box.tracking_id)
            track_point_clouds[track_key].append(points_masked)

    for track_key, point_cloud_list in track_point_clouds.items():
        sequence_token, tracking_id = track_key
        save_path = os.path.join(
            save_dir, f"point_clouds_{sequence_token}_{tracking_id}.npy"
        )

        # Process each point cloud in the list
        processed_point_cloud_list = []
        for pc in point_cloud_list:
            num_points = len(pc)
            if num_points > 0:
                # Sample N_max points from the point cloud
                idx = np.random.choice(num_points, N_max, replace=True)
                sampled_pc = pc[idx]
            else:
                # Pad the point clouds with zeros to have the same number of points (N_max)
                sampled_pc = np.zeros((N_max, 3))
            processed_point_cloud_list.append(sampled_pc)

        # Reshape the final combined point cloud array to have the shape (T, N_max, 3)
        T = len(processed_point_cloud_list)
        combined_point_cloud = np.stack(processed_point_cloud_list, axis=0).reshape(
            T, N_max, 3
        )

        np.save(save_path, combined_point_cloud)
        print("Saved to", save_path)


def preprocess_and_save_point_clouds(tracking_results, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get all the sequences
    sequences = tracking_results.seq_tokens

    N_max = 1000

    with multiprocessing.Pool(processes=10) as pool:
        results = [
            pool.apply_async(
                process_sequence, (sequence_token, tracking_results, N_max, save_dir)
            )
            for sequence_token in sequences
        ]

        for result in tqdm.tqdm(results, total=len(results), desc="Processing PC:"):
            try:
                result.get()
            except Exception as e:
                print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="/datasets/zod/zodv2",
        help="Path to raw dataset",
    )
    parser.add_argument(
        "--version", type=str, default="full", help="one of [full, mini]"
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out-dir", type=str, default="/storage/preprocessed")
    parser.add_argument(
        "--tracking-result-path",
        type=str,
        default="/home/s0001671/workspace/storage/results/nuscenes-centerpoint-valsplit-trackresults.json",
    )
    parser.add_argument("--verbose", type=bool, default=True)

    args = parser.parse_args()

    preprocess(
        args.data_path,
        args.version,
        args.split,
        args.out_dir,
        args.tracking_result_path,
    )
