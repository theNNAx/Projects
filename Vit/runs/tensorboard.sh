#! /bin/bash
latest_file=$(ls -t | head -n 1)

latest_path_file=$(dirname "$latest_file")

cd "$latest_file"

tensorboard --logdir="$latest_path_file" --port=4004
