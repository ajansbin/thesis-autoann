import json
import yaml


def load_config(config_path: str):
    with open(config_path) as f:
        file_extension = config_path.split(".")[-1]
        if file_extension == "json":
            conf = json.load(f)
        elif file_extension in ["yaml", "yml"]:
            conf = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            raise ValueError(
                f"Invalid file format: {file_extension}. Must be either JSON or YAML."
            )
    return conf
