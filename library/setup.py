from setuptools import setup

setup(
    name='ai_tools',
    version='1.0',
    description='Custom tools for training models with Tensorflow.',
    author='Andrea Spiteri',
    author_email='andrea.spiteri19@gmail.com',
    packages=['ai_tools'],
    install_requires=[
        'wheel',
        'pandas',
        'tensorflow',
        'numpy',
        'matplotlib',
        'sklearn',
        'aim'
    ]
)

# setup(
#     name='utils',
#     version='1.0',
#     description='Custom utilities',
#     author='Andrea Spiteri',
#     author_email='andrea.spiteri19@gmail.com',
#     packages=['utils'],
#     install_requires=[
#         'wheel',
#         'pandas',
#         'tensorflow',
#         'numpy',
#         'matplotlib',
#         'sklearn',
#         'librosa',
#         'ipython',
#         'soundfile',
#         'yaml'
#     ],
# )
