

from visualizer.zod.track_data import ZodTrackSequence

def main(data_path:str, track_results:str, save_dir:str, seq_id:str, nr_frames:int):
    track_seq = ZodTrackSequence(data_path, track_results, save_dir, seq_id)
    track_seq.create_bev_animation(nr_frames)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/datasets/zod/zodv2", help="Path to raw dataset")
    parser.add_argument('--track-results', type=str, default='/staging/agp/masterthesis/2023autoann/storage/tracking/SimpleTrack_zod_full_val/results/results.json')
    parser.add_argument('--seq-id', type=str, default="000002")
    parser.add_argument('--nr-frames', type=int, default="10")
    parser.add_argument('--save-dir', type=str, default="/home/s0001668/workspace/storage/tracking/visualized/")

    args = parser.parse_args()

    main(args.data_path, args.track_results, args.save_dir, args.seq_id, args.nr_frames)