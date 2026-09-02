import json
from pathlib import Path

CONFIG_FILE = 'config.json'
EXAMPLE_FILE = 'config.example.json'

DIRECTORY_KEYS = ['trainDirectory', 'trackDirectory', 'journeyDirectory']


def loadConfig():

    """
    Load the local input configuration of the multi-stop wrappers.

    The configuration lives in config.json next to this module and is not tracked by git,
    so that local data paths and train/track/journey ids stay private.
    Copy config.example.json to config.json and adapt it to your setup.

    The keys in DIRECTORY_KEYS default to the value of "directory" if not given.
    """

    configPath = Path(__file__).parent / CONFIG_FILE

    assert configPath.is_file(), (
        f"Missing configuration file '{configPath}'. "
        f"Copy '{EXAMPLE_FILE}' to '{CONFIG_FILE}' and adapt it to your setup."
    )

    with open(configPath) as f:
        config = json.load(f)

    assert 'directory' in config, f"Key 'directory' is missing in '{configPath}'"

    for key in DIRECTORY_KEYS:
        config.setdefault(key, config['directory'])

    return config
