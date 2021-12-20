# Extends an exisiting docker container by installing some additional dependecies specific for audio AI.
# Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu-jupyter.Dockerfile
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Python
RUN pip install librosa pandas opencv-python scikit-image pydub torchaudio kapre sox

RUN apt-get update
RUN apt-get install -y sox libsox-fmt-mp3 ffmpeg libsm6 libxext6
