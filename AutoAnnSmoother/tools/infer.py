from tools.utils.smoothing_inferer import SmoothingInferer

def main(config:str, data_path:str, save_dir:str, track_results:str, model_path:str, seq_id:str):
    model = SmoothingInferer(track_results, config, data_path, save_dir, seq_id)
    model.load_data()
    model.load_model(model_path)
    model.infer()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/training_config.yaml", help="Path to config to use")
    parser.add_argument('--data-path', type=str, default="/datasets/nuscenes/v1.0", help="Path to raw dataset")
    parser.add_argument('--save-dir', type=str, default="/staging/agp/masterthesis/2023autoann/storage/smoothing/inference")
    parser.add_argument('--track-results', type=str, default="/staging/agp/masterthesis/2023autoann/storage/tracking/NuSC-transfusion-simpletrack/nuScenes/2hz/results/SimpleTrack2Hz/results/nuscenes-centerpoint-valsplit-trackresults.json")
    parser.add_argument('--model', type=str, default="/staging/agp/masterthesis/2023autoann/storage/smoothing/test/pointnet/pointnet_nuscenes_val_model.pth")
    parser.add_argument('--seq-id', type=str, default="0000000", help="sequence id to infere on")

    args = parser.parse_args()

    main(args.config, args.data_path, args.save_dir, args.track_results, args.model, args.seq_id)