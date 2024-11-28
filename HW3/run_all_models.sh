#!/bin/bash

models=("Net" "CNN")
batch_sizes=(16 64 256)
lrs=(0.1 0.01 0.001)
epochs=(10 100 500)

for model in "${models[@]}"; do
  for batch_size in "${batch_sizes[@]}"; do
    for lr in "${lrs[@]}"; do
      for epoch in "${epochs[@]}"; do
        echo "Training $model with batch size $batch_size, learning rate $lr, and $epoch epochs"
        python main.py --model $model --batch-size $batch_size --lr $lr --epochs $epoch
      done
    done
  done
done
