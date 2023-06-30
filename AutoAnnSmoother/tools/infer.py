from tools.utils.smoothing_inferer import SmoothingInferer


def main(
    config: str,
    data_path: str,
    pc_name: str,
    save_path: str,
    result_path: str,
    model_path: str,
    version: str,
    split: str,
):
    model = SmoothingInferer(
        result_path, config, data_path, version, split, pc_name, save_path
    )
    model.load_data()
    model.load_model(model_path)
    model.infer(N=15)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path",
        type=str,
        default="result.json",
    )
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
    parser.add_argument(
        "--pc-name",
        type=str,
        default="preprocessed_full_train",
        help="Name of folder where pc are stored",
    )
    parser.add_argument("--save-path", type=str, default="/storage/train")
    parser.add_argument("--model-path", type=str, default="run1", help="Model run name")
    parser.add_argument("--version", default="mini", choices=["full", "mini"])
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])

    args = parser.parse_args()

    main(
        args.config,
        args.data_path,
        args.pc_name,
        args.save_path,
        args.result_path,
        args.model_path,
        args.version,
        args.split,
    )
