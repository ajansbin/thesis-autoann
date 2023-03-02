import yaml
from yaml.loader import SafeLoader


def load_config(config_path:str):
    with open(config_path) as f:
        conf = yaml.load(f, Loader=SafeLoader)
    return conf