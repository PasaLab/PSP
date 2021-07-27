#!/bin/bash

log_folder="./evaluate_log_output/log_output"
train_his_folder="./evaluate_log_output/train_history"
if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

arg=$1
cur_time="`date +%Y-%m-%d-%H-%M-%S`"
train_info_dir="$train_his_folder/$arg-$cur_time"
SPLIT_TYPE=$2

if [ ! -x train_info_dir ]; then
mkdir -p $train_info_dir
fi
log_file="$log_folder/$arg-$cur_time.log"
python_command="python3 evaluate.py --dataset $arg --split_type ${SPLIT_TYPE} --logger_path $train_info_dir 2>&1"
log_command="tee -i $log_file"
echo "Current time: $cur_time"
echo "Run command: $python_command"
echo "Log info into file: $log_file"
echo "Train info file: $train_info_dir"
eval "$python_command | $log_command"


