

from visualizer.zod.track_data import ZodTrackSequence

def main(data_path:str, track_results:str, save_dir:str, tag:str, seq_ids:str, nr_frames:int, no_tracks:bool, gt:bool):
    
    seq_ids = [str(item) for item in seq_ids.split(',')]
    for seq_id in seq_ids:
        print('SEQ', seq_id)
        name = str(seq_id) if tag=='' else str(seq_id) + '_' + tag
        track_seq = ZodTrackSequence(data_path, track_results, save_dir, name, seq_id, no_tracks, gt)
        track_seq.create_lidar_animation(nr_frames)

    #track_seq.bev_detections(nr_frames)
    #track_seq.bev_detections_animations(nr_frames)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default="/datasets/zod/zodv2", help="Path to raw dataset")
    parser.add_argument('--track-results', type=str, default='/staging/agp/masterthesis/2023autoann/storage/tracking/SimpleTrack_zod_full_val/results/results.json')
    parser.add_argument('--seq-id', type=str, default="000002")
    parser.add_argument('--nr-frames', type=int, default="10")
    parser.add_argument('--save-dir', type=str, default="/home/s0001668/workspace/storage/tracking/visualized/full_train_subset_2")
    parser.add_argument('--no-tracks', action='store_false')
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--gt', action='store_true', help='show gt on results')
    
    args = parser.parse_args()

    main(args.data_path, args.track_results, args.save_dir, args.tag, args.seq_id, args.nr_frames, args.no_tracks, args.gt)