import os
import yaml

this_file = os.path.abspath(__file__)
this_dir = os.path.split(this_file)[0]

PROJECT_ROOT = os.path.abspath(os.path.join(this_dir, '..'))

user_settings = None

SETTINGS_FILE = os.path.join(PROJECT_ROOT, 'sets.yaml')

if os.path.exists(os.path.join(PROJECT_ROOT, 'sets.yaml')):
    with open(SETTINGS_FILE, 'r') as fp:
        user_settings = yaml.safe_load(fp)
