from tools.utils.smoothing_trainer import SmoothingTrainer

def main(config:str, data_path:str, save_dir:str, result_path:str):
    trainer = SmoothingTrainer(result_path, config, data_path, save_dir)
    trainer.load_data()
    trainer.load_model()
    trainer.train()
    trainer.save_results()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configs/training_config.yaml", help="Path to config to use")
    parser.add_argument('--data-path', type=str, default="/datasets/nuscenes/v1.0", help="Path to raw dataset")
    parser.add_argument('--save-dir', type=str, default="/storage/train")
    parser.add_argument('--result-path', type=str, default="/home/s0001671/workspace/storage/results/nuscenes-centerpoint-valsplit-trackresults.json")

    args = parser.parse_args()

    main(args.config, args.data_path, args.save_dir, args.result_path)