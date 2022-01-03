import sys

sys.path.append('../library/')  # Add custom library to PYTHONPATH.

from ai_tools import DataGenerator
from ai_tools.models import build_conv2d_example
from tensorflow.keras.models import Model
from typing import List
from utils.helpers import load_data, Data

# Create train and validation generators.
train_data_generator, val_data_generator = DataGenerator.from_path_to_audio(
    '../data-sets/small_nsynth/train',
    batch_size=124
)

# Build model.
model: Model = build_conv2d_example()

# Train model.
history = model.fit(
    train_data_generator,
    validation_data=val_data_generator,
    epochs=5
)

model.save('../models/simple_nsynth_classification')

# X_test, y_test = load_data('../data-sets/small_nsynth/test')

# model.evaluate(X_test, y_test)

