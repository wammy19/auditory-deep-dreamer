"""
The settings from ../config.yml are read and set in this file so that they can be accessed globally.
"""

import yaml

with open('../config.yml') as file_handler:
    service_config: dict = yaml.safe_load(file_handler)

dataset_path: str = service_config['dataset_path']
aim_logs_path: str = service_config['aim_logs_path']
model_checkpoint_path: str = service_config['model_checkpoint_path']
model_config_csv_log_path: str = service_config['model_config_csv_log_path']
model_evaluation_csv_log_path: str = service_config['model_evaluation_csv_log_path']
logs_path: str = service_config['logs_path']
model_histories: str = service_config['model_histories']

del service_config
