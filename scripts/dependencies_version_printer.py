"""
Check that dependencies are working correctly with conda env.
"""

import librosa
import tensorflow
import kapre
import numpy
import sklearn
import scipy


def main():
    print(f'Tensorflow {tensorflow.__version__}')
    print(f'Kapre {kapre.__version__}')
    print(f'librosa {librosa.__version__}')
    print(f'numpy {numpy.__version__}')
    print(f'sklearn {sklearn.__version__}')
    print(f'scipy {scipy.__version__}')


if __name__ == '__main__':
    main()
