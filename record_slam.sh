#!/bin/bash
OUTPUT_VIDEO=$1
SLAM_CMD=$2
SLAM_ARGS=$3

echo "Starting Xvfb..."
Xvfb :99 -screen 0 1024x768x24 &
XVFB_PID=$!
sleep 2

echo "Starting ffmpeg recording to $OUTPUT_VIDEO.mkv..."
ffmpeg -video_size 1024x768 -framerate 30 -f x11grab -i :99.0+0,0 -c:v libx264 -preset ultrafast -y $OUTPUT_VIDEO.mkv &
FFMPEG_PID=$!

echo "Running SLAM command..."
export DISPLAY=:99
$SLAM_CMD $SLAM_ARGS

echo "SLAM command finished. Stopping recording..."
kill -INT $FFMPEG_PID
sleep 3
kill -9 $FFMPEG_PID
kill -9 $XVFB_PID
ffmpeg -i $OUTPUT_VIDEO.mkv -c copy -y $OUTPUT_VIDEO
rm $OUTPUT_VIDEO.mkv
echo "Done."
