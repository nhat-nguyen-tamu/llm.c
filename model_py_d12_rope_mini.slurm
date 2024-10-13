#!/bin/bash
#SBATCH --job-name=gpt2_train_py_d12_rope_mini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00              #Request 10 minutes
#SBATCH --mem=128GB                  #Request 128GB per node
#SBATCH --partition=gpu              #Request the GPU partition/queue
#SBATCH --gres=gpu:a100:1            #Request one A100 GPU to use
#SBATCH --output=gpt2_train.%j.log   #Redirect stdout/err to file

# Run the training script
python train_gpt2.py \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "dev/data/fineweb10B/fineweb_val_*.bin" \
    --model "d12" \
    --output_dir "log_model_py_d12_rope_mini" \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --zero_stage 1 \
    --weight_decay 0.1 \
    --learning_rate 0.0006 \
    --learning_rate_decay_frac 0.0 \
    --warmup_iters 700 \
    --val_loss_every 250 \
    --sample_every 20000 \
    --flash 1 \
    --dtype "bfloat16" \
    --num_iterations 125