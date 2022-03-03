#!/bin/sh

docker build -f ./docker-files/Dockerfile.jupyter --rm -t aspit002/audio-deepdream-dev-env-jupyter .
docker build -f ./docker-files/Dockerfile.py_runner --rm -t aspit002/audio-deepdream-dev-env .
docker build -f ./docker-files/Dockerfile.aim --rm -t aspit002/aim-logs .
