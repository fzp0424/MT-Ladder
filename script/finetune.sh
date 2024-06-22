num_epochs=1
per_device_train_batch_size=1
per_device_eval_batch_size=1
gradient_accumulation_steps=4
max_seq_length=512
lr=1e-4
lora_rank=16
lora_alpha=16
lora_dropout=0.10
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj,lm_head"
modules_to_save=None
pretrained_model="google/gemma-2b"
train_file="../dataset/train/Mixed/Mixed.json"
output_dir="<<<Your Model Output Path>>>"
data_cache_dir="<<<Your Data Cache Path>>>"
validation_file=None
deepspeed_config_file="../configs/ds_zero2_no_offload_bf16.json"
padding_side="left"
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
seed=20424

save_strategy="no"
prompt_strategy="intermediate"
use_shots=True

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_OF_USING_GPUS=4

echo "Base model: {$pretrained_model}"
echo "Output dir: {$output_dir}"
echo "train_file: {$train_file}"
echo "deepspeed config: {$deepspeed_config_file}"
echo "max_seq_length: {$max_seq_length}"

torchrun --nnodes 1 --nproc_per_node ${NUM_OF_USING_GPUS} --master_port ${port} \
    ../peft_finetune.py \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name_or_path ${pretrained_model} \
    --deepspeed ${deepspeed_config_file} \
    --padding_side ${padding_side} \
    --prompt_strategy ${prompt_strategy} \
    --use_shots ${use_shots} \
    --torch_dtype "auto" \
    --bf16 \
    --do_train \
    --do_eval False \
    --use_wandb False \
    --train_file ${train_file} \
    --validation_file ${validation_file} \
    --data_cache_dir ${data_cache_dir} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_seq_length ${max_seq_length} \
    --num_train_epochs ${num_epochs} \
    --lr_scheduler_type cosine \
    --optim "adamw_torch" \
    --learning_rate ${lr} \
    --warmup_ratio 0.01 \
    --weight_decay 0 \
    --logging_first_step True \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy ${save_strategy} \
    --evaluation_strategy no \
    --eval_steps 100 \
    --save_steps 0.1 \
    --preprocessing_num_workers 8 \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --seed ${seed} \
    --ddp_find_unused_parameters False \
    --ddp_timeout 30000 








