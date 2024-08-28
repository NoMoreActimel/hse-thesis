import json
import logging
import logging.config

from collections import OrderedDict
from pathlib import Path

def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)

def setup_logging(
        save_dir, log_config=None, root_path=None, default_level=logging.INFO
):
    if log_config is None:
        log_config = str(root_path / "logger" / "logger_config.json")
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)

def get_logger(
        
)