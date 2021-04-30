#!/usr/bin/bash

python reinforce.py \
       --alice_model_file rnn_model.th \
       --bob_model_file rnn_model.th \
       --joe_model_file rnn_model.th \
       --output_model_file rnn_rl_model.th \
       --context_file data/negotiate/selfplay.txt  \
       --temperature 0.5 \
       --log_file rnn_rl.log \
       --sv_train_freq 4 \
       --nepoch 1 \
       --selection_model_file selection_model.th  \
       --rl_lr 0.00001 \
       --rl_clip 0.0001 \
       --sep_sel \
       --ctx_num 100 \
       --engine_train \
       --delta 0.005 \
       --nu 0.5
