#!/bin/bash

# Recursively loop through directories and convert mp3's into wav files using sox.

# Dependencies:
# sudo apt install -y sox libsox-fmt-mp3

find . -name "*.mp3" -type f -exec sox {} "$(basename {} .mp3).wav" \; -exec rm {} \;
