"""
Check that dependencies are working correctly within conda env.
"""

import librosa
import tensorflow
import kapre
import numpy
import sklearn
import scipy

print(tensorflow.__version__)
print(kapre.__version__)
print(librosa.__version__)
print(numpy.__version__)
print(sklearn.__version__)
print(scipy.__version__)
