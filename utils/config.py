import json
import yaml
from bunch import Bunch
import os, sys
import logging

def get_config_from_json(json_file: str):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def get_config_from_yaml(yaml_file: str):
    """
    Get the config from a yaml file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    logging.debug("Looking for config file at {}".format(yaml_file))

    if os.path.exists(yaml_file):
        logging.info("Found config file at {}".format(yaml_file))
    else:
        logging.critical("Did not find config file at {}".format(yaml_file))
        logging.critical("Exiting program.")
        sys.exit(0)

    # parse the configurations from the config json file provided
    with open(yaml_file, 'r') as stream:
        try: 
            config_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            config_dict = {None: None}
            logging.critical(exc)
            logging.critical("Could not load config file at {}".format(yaml_file))
            logging.critical("Exiting program.")
            sys.exit(0)
    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    
    return config, config_dict


def process_config(file: str):
    config, _ = get_config_from_yaml(file)
    logging.debug("config: ", config)
    logging.debug("config dictionary: ", _)
    return config