#!/bin/bash

# Scripts created to convert the philharmonia instrument data-set from mp3 format to wav files so liberos can load them.

# Dependencies:
# sox

# Learning resources:
# 1. https://unix.stackexchange.com/questions/171677/cd-into-all-directories-execute-command-on-files-in-that-directory-and-return/171679
# 2. https://unix.stackexchange.com/questions/425532/batch-convert-mp3-files-to-wav-using-sox

for dir in ./*; do
  if [ -d "$dir" ]; then
    cd "$dir"
    for subDir in ./*; do
      if [ -d "$subDir" ]; then
        cd "$subDir"
        for i in *.mp3; do
          sox "$i" "$(basename "$i" .mp3).wav"
          rm "$i"
        done
        cd ..
      fi
    done
    cd ..
  fi
done
