#!/bin/bash

R_DIR=$(dirname $0)
MYDIR=$(
  cd $R_DIR
  pwd
)
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_sync_nccl_allreduce=1
export PYTHONPATH=/home/Model/ernie:${PYTHONPATH:-}
export TASK_DATA_PATH=/home/Model/data/MAMS-ATSA/processed
export MODEL_PATH=/home/Model/ERNIE_Base_en_stable-2.0.0

if [[ -f ./model_conf ]]; then
  source ./model_conf
else
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

mkdir -p log/

lr=3e-5
batch_size=32
epoch=4

for i in {1..5}; do
  timestamp=$(date "+%Y-%m-%d-%H-%M-%S")
  python3 -u /home/Model/ernie/run_classifier.py \
  --for_cn False \
  --task 'term' \
  --use_cuda True \
  --use_fast_executor ${e_executor:-"true"} \
  --tokenizer ${TOKENIZER:-"FullTokenizer"} \
  --use_fp16 ${USE_FP16:-"false"} \
  --do_train true \
  --do_val true \
  --do_test true \
  --batch_size $batch_size \
  --init_pretraining_params ${MODEL_PATH}/params \
  --verbose true \
  --train_set ${TASK_DATA_PATH}/train.npz \
  --dev_set ${TASK_DATA_PATH}/dev.npz,${TASK_DATA_PATH}/test_label.npz \
  --test_set ${TASK_DATA_PATH}/test_no.npz \
  --vocab_path ${MODEL_PATH}/vocab.txt \
  --checkpoints ./checkpoints \
  --save_steps 10000 \
  --weight_decay 0.0 \
  --warmup_proportion 0.1 \
  --validation_steps 100000000000 \
  --epoch $epoch \
  --max_seq_len 128 \
  --ernie_config_path ${MODEL_PATH}/ernie_config.json \
  --learning_rate $lr \
  --skip_steps 10 \
  --num_iteration_per_drop_scope 1 \
  --num_labels 3 \
  --metric 'acc_and_f1' \
  --test_save output/test_out.$i.$lr.$batch_size.$epoch.csv \
  --random_seed 1 2>&1 | tee log/job.$i.$lr.$batch_size.$epoch.log \

done
