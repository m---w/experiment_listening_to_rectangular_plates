#!/bin/bash

for i in *.wav;
  do name=`echo "$i" | cut -d'.' -f1`
  echo "$name"
  ffmpeg -y -i "$i" -ac 2 -vn -b:a 320k -cutoff:a 22000 "${name}.mp3"
done