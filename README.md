# **Auditory DeepDream**

### Downloading datasets

Small data set (≈4.6gb):

`wget https://www.dropbox.com/s/v6hbvfwuyjm2ajp/small_nsynth.zip && unzip small_nsynth.zip`

### Running with Docker

_Note about GPU usage:_

The quickest way to get Tensorflow running on GPU is with a docker container as
suggested [here](https://www.tensorflow.org/install/docker). The following are links to the required software and
instructions on how to install them.

Requirements for GPU:

- [Nvidia Container Tools](https://github.com/NVIDIA/nvidia-docker)
- [Nvidia Drivers](https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu)
- [Cuda Tool Kit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Build docker image for developer environment utilizing jupyter-notebook:

`docker build -f ./docker-files/Dockerfile.jupyter --rm -t aspit002/audio-deepdream-dev-env-jupyter .`

Build docker image for running `./src/main.py`:

`docker build -f ./docker-files/Dockerfile.py_runner --rm -t aspit002/audio-deepdream-dev-env .`

Before spinning up the containers, have a look at the docker-compose.yml file. If your machine doesn't have an Nvidia
GPU, you will need to comment out the devices being passed into the container. Also set the memory for the container
appropriately. It's currently set to 20gb because I have a lot of memory on machine.

You also won't be able to run any notebooks that require a dataset as they are not available in the repo. In the future,
I might create downloader scripts so this isn't an issue.

To start up the Jupyter server run:

`docker-compose up -d && sudo docker-compose logs -f`

Then follow the link with token, example link:

`http://127.0.0.1:8888/?token=68fea5fe14cb2a97bb0fb016f0690b91b664e8df9f75436d`

To stop the Jupyter server run:

`docker-compose kill`

_Note: Tensorflow's Docker image uses Python3.8_
