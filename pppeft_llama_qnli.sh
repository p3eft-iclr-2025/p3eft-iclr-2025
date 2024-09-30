export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export n_of_loras=1
export batch_size=48
export coefs_method_name="averaged"
export activation_lr_rw=0.0
export shift_lr_rw=0.0
export activation_dc_rw=0.0
export shift_dc_rw=0.0
export mult_std=0
export lora_rank=16
export lora_alpha=32
export lora_dropout=0.0
export max_seq_length=512
export lr=2e-4
export lr_scheduler_type="linear"
export model_name="Llama-2-7b-hf"
export bf16=true
export n_epoch=5
export seed=0
export loras_gradient_checkpointing=false
export model_gradient_checkpointing=false
export output_dir="./llama2_qnli"
export num_gpus=8
torchrun --nproc_per_node=$num_gpus --standalone \
src/main.py \
--model_name_or_path meta-llama/$model_name \
--task_name qnli \
--do_train \
--do_eval \
--max_seq_length $max_seq_length \
--per_device_train_batch_size $(( $batch_size / $num_gpus )) \
--per_device_eval_batch_size 64 \
--learning_rate $lr \
--num_train_epochs $n_epoch \
--output_dir $output_dir/model \
--bf16 $bf16 \
--gradient_accumulation_steps 1 \
--evaluation_strategy steps \
--eval_steps 250 \
--save_strategy no \
--warmup_ratio 0.03 \
--seed $seed \
--weight_decay 0.01 \
--logging_steps 10 \
--logging_dir $output_dir/log \
--n_of_loras $n_of_loras \
--lora_rank $lora_rank \
--lora_alpha $lora_alpha \
--lora_dropout $lora_dropout \
--mult_std $mult_std \
--coefs_method_name $coefs_method_name \
--activation_lr_rw $activation_lr_rw \
--shift_lr_rw $shift_lr_rw \
--activation_dc_rw $activation_dc_rw \
--shift_dc_rw $shift_dc_rw \
--lr_scheduler_type $lr_scheduler_type \
--loras_gradient_checkpointing $loras_gradient_checkpointing \
--model_gradient_checkpointing $model_gradient_checkpointing \
--ddp_find_unused_parameters false \
--report_to none \