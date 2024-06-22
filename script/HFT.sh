train_directory="../dataset/train/HFT" 
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
pretrained_model="google/${found_model}"
validation_file=None
peft_path=None
deepspeed_config_file="../configs/ds_zero2_no_offload_bf16.json"
padding_side="left"
port=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
seed=20424

save_strategy="no"
found_model="gemma-2b"
prompt_strategy="intermediate"
use_shots=True

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_OF_USING_GPUS=4

for train_file in "$train_directory"/*; do

    if [ -f "$train_file" ]; then

        only_train_file=$(basename -- "$train_file")
        filename=$(basename -- "$only_train_file")
        filename_no_extension="${filename%.*}"

        output_dir="<<<your ckpt path>>>/${filename_no_extension}"
        data_cache_dir="<<<your cache path>>>/$only_train_file"
        
        echo "Base model: {$pretrained_model}"
        echo "train_file: {$train_file}"
        echo "PEFT Path: {$peft_path}"
        echo "Output dir: {$output_dir}"


        torchrun --nnodes 1 --nproc_per_node ${NUM_OF_USING_GPUS} --master_port ${port} \
            ../peft_finetune.py \
            --peft_path ${peft_path} \
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
            --use_wandb False\
            --train_file ${train_file} \
            --data_cache_dir ${data_cache_dir} \
            --validation_file ${validation_file} \
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

        peft_path="${output_dir}"

    fi
done
