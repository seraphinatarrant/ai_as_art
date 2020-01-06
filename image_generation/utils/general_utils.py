import yaml

def read_yaml_config(config_file):
    return yaml.load(open(config_file), Loader=yaml.FullLoader)
