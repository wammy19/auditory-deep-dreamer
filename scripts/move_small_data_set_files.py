import os
import shutil

reed = os.listdir('../data-sets/small_nsynth/train/string')
print(f'Original size: {len(reed)}')
subset_of_reed = reed[:len(reed) // 10]

for inst in subset_of_reed:
    shutil.move(f'../data-sets/small_nsynth/train/string/{inst}', '../data-sets/small_nsynth/validation/string')

reed = os.listdir('../data-sets/small_nsynth/train/string')
print(f'New size: {len(reed)}')
