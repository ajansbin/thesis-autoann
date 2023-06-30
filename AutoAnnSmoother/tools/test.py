# ToDo
from tools.utils.smoothing_tester import SmoothingTester


def main(
    config: str, data_path: str, save_dir: str, track_results: str, model_path: str
):
    model = SmoothingTester(track_results, config, data_path, save_dir)
    model.load_data()
    model.load_model(model_path)
    model.test()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to config to use",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/datasets/nuscenes/v1.0",
        help="Path to raw dataset",
    )
    parser.add_argument("--save-dir", type=str, default="/out-dir")
    parser.add_argument(
        "--track-results", type=str, default="/path/to/trackresults.json"
    )
    parser.add_argument("--model", type=str, default="/path/to/model.pth")

    args = parser.parse_args()

    main(args.config, args.data_path, args.save_dir, args.track_results, args.model)
