#!/bin/bash

log_folder="./evaluate_log_output/log_output"
train_his_folder="./evaluate_log_putput/train_history"
if [ ! -x $log_folder ]; then
	mkdir $log_folder
fi

arg=$1
# dataset_dir="/home/zhuruancheng/wwj_space/AutoGNN/public/$arg"
split_type=$2
is_use_early_stop=$3
NUM_CELLS=$4
NUM_NODES=$5

# split_ids="1 2 3 4 5 6 7 8 9" 
split_ids="0" 

for SPLIT_ID in `echo ${split_ids}`; do
    cur_time="`date +%Y-%m-%d-%H-%M-%S`"
    train_info_dir="$train_his_folder/$arg-${SPLIT_ID}-$cur_time"

    if [ ! -x train_info_dir ]; then
    mkdir -p $train_info_dir
    fi
    log_file="$log_folder/$arg-${SPLIT_ID}-$cur_time.log"
    python_command="python evaluate.py --dataset $arg --split_id ${SPLIT_ID} --use_early_stop $is_use_early_stop --num_cells ${NUM_CELLS} --num_nodes ${NUM_NODES} --logger_path $train_info_dir 2>&1"
    log_command="tee -i $log_file"
    echo "Current time: $cur_time"
    echo "Run command: $python_command"
    echo "Log info into file: $log_file"
    echo "Train info file: $train_info_dir"
    eval "$python_command | $log_command"
done




