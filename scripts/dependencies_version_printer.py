"""
Check that dependencies are working correctly within conda env.
"""

import librosa
import tensorflow
import kapre
import numpy
import sklearn
import scipy

print(f'Tensorflow {tensorflow.__version__}')
print(f'Kapre {kapre.__version__}')
print(f'librosa {librosa.__version__}')
print(f'numpy {numpy.__version__}')
print(f'sklearn {sklearn.__version__}')
print(f'scipy {scipy.__version__}')
