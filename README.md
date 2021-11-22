# **Auditory DeepDream**

### Downloading datasets.

Yet to come...

### Running jupyter-notebooks with docker.

The quickest way to get Tensorflow running on GPU is with a docker container as
suggested [here](https://www.tensorflow.org/install/docker). The following are links to the required software and
instructions on how to install them.

Requirements:

- [Docker](https://docs.docker.com/engine/install/ubuntu/)
- [Nvidia Container Tools](https://github.com/NVIDIA/nvidia-docker)
- [Nvidia Drivers](https://phoenixnap.com/kb/install-nvidia-drivers-ubuntu)
- [Cuda Tool Kit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Build docker image for developer environment:

`docker build . -t aspit002/audio-deepdream-dev-env`

To start up the Jupyter server run:

`docker-compose up -d && sudo docker-compose logs -f`

Then follow the link with token, example link:

`http://127.0.0.1:8888/?token=68fea5fe14cb2a97bb0fb016f0690b91b664e8df9f75436d`

To stop the Jupyter server run:

`docker-compose kill`

_Note: Tensorflow's Docker image uses Python3.8_
