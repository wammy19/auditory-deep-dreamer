import os
import re
import shutil

patter_for_instrument: re.Pattern = re.compile(r'^.*?(?=_)')

path_to_data_sets: str = '../data-sets/nsynth/validaiton'

for file in os.listdir(path_to_data_sets):

    if os.path.isdir(os.path.join(path_to_data_sets, file)):  # Ignore directories.
        continue

    instrument: str = re.findall(patter_for_instrument, file)[0]
    instrument_dir: str = os.path.join(path_to_data_sets, instrument)

    if os.path.isdir(instrument_dir) is False:  # If the class directory doesn't exist, create it.
        os.makedirs(instrument_dir)

    shutil.move(os.path.join(path_to_data_sets, file), instrument_dir)  # Move file into appropriate folder.
