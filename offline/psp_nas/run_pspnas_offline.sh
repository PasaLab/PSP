#!/bin/bash
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

log_folder="./log_output"
train_his_folder="./train_history"
if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

arg=$1
cur_time="`date +%Y-%m-%d-%H-%M-%S`"
train_info_dir="$train_his_folder/$arg-$cur_time"
is_use_early_stop=$2
SPLIT_TYPE=$3

if [ ! -x train_info_dir ]; then
mkdir -p $train_info_dir
fi
log_file="$log_folder/$arg-$cur_time.log"
python_command="python3 -W ignore search_gbdtnas.py --dataset $arg --use_early_stop $is_use_early_stop --split_type ${SPLIT_TYPE} --logger_path $train_info_dir 2>&1"
log_command="tee -i $log_file"
echo "Current time: $cur_time"
echo "Run command: $python_command"
echo "Log info into file: $log_file"
echo "Train info file: $train_info_dir"
eval "$python_command | $log_command"

