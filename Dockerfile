# Extends an exisiting docker container by installing some editional dependecies.
# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN pip install librosa torchaudio