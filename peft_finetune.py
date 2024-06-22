import logging
import os
import sys
import torch
import transformers
import pandas as pd
import datasets

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from peft import (
    LoraConfig,
    AdaLoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    set_peft_model_state_dict,
    get_peft_model_state_dict
)

from utils.arguments import ModelArguments, DataTrainingArguments, MyTrainingArguments
from utils.build_dataset import build_instruction_dataset, DataCollatorForSupervisedDataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

class SavePeftModelCallBack(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint)
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        
        peft_model_path = os.path.join(checkpoint_folder)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir)
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)
        

logger = logging.getLogger(__name__)

def main():
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        #If we pass only one argument to the script and it's path to a json file
        #let's parse it to get our arguments
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    #setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if training_args.should_log:
        #The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    #log on each process the small summary
    logger.warning(
        f"Process rank:{training_args.local_rank}, device{training_args.device}, n_gpu:{training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, fp 16-bits training: {training_args.fp16}, bf 16-bits training: {training_args.bf16}"
    )

    # Only overwrite environ if wandb param passed
    if training_args.use_wandb:
        os.environ["WANDB_DIR"] = training_args.output_dir
        if len(training_args.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = training_args.wandb_project
        if len(training_args.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = training_args.wandb_log_model
    else:
        os.environ['WANDB_MODE'] = 'disabled'

    #set seed before initializing model
    set_seed(training_args.seed)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None
    }

    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf = bool(".ckpt" in model_args.model_name_or_path),
        config = model_config,
        torch_dtype = torch_dtype,
        low_cpu_mem_usage = True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)

    if tokenizer.pad_token is None:
        if "llama3" in model_args.model_name_or_path:
            tokenizer.pad_token_id = 128002  #make sure modify Llama xx_config.json first
            tokenizer.pad_token = "<|pad_of_text|>" #make sure modify Llama xx_config.json first
            logger.info(f"set pad token id :{tokenizer.pad_token_id}")
        else:
            tokenizer.add_special_tokens(dict(pad_token = DEFAULT_PAD_TOKEN))
            logger.info(f"set pad token id :{tokenizer.pad_token_id}")
            
    logger.info(f"pad_token:{tokenizer.pad_token} pad_token_id: {tokenizer.pad_token_id}")

    if model_args.padding_side:
        tokenizer.padding_side = model_args.padding_side
        logger.info(f"padding side:{tokenizer.padding_side}")
    
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"embedding_size:{embedding_size}")

    if len(tokenizer) != embedding_size:
        logger.info("Resize the embedding size by the size of the tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    
    #continue training or init new peft model
    if training_args.peft_path is not None and training_args.peft_path != "None" and not training_args.continue_finetune:
        logger.info("Peft from pre-trained model")
        model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        print("From {} successfully load lora checkpoint".format(training_args.peft_path))
    else:
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(',')
        modules_to_save = training_args.modules_to_save
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(',')
        lora_dropout = training_args.lora_dropout
        lora_alpha = training_args.lora_alpha
        logger.info(f"target_modules: {target_modules}")
        logger.info(f"modules_to_save: {modules_to_save}")
        
        if not training_args.use_adalora:
            lora_rank = training_args.lora_rank
            logger.info(f"lora_rank: {lora_rank}")
            peft_config = LoraConfig(
                use_dora=training_args.use_dora,
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r = lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
        else:
            init_rank = training_args.init_rank
            target_rank = training_args.target_rank
            logger.info(f"init_rank: {init_rank}")
            logger.info(f"target_rank: {target_rank}")
            peft_config = AdaLoraConfig(
                peft_type="ADALORA",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                init_r = init_rank,
                target_r = target_rank, 
                inference_mode=False,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                modules_to_save=modules_to_save)
            
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    if data_args.train_file is not None:
        train_data_path = data_args.train_file
    else:
        raise ValueError(f"Please check the train_file path!")

    if data_args.validation_file is not None:
        val_data_path = data_args.validation_file
    else:
        raise ValueError(f"Please check the val_file path!")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset=None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc = "loading and tokenization"):
            train_dataset = build_instruction_dataset(
                data_path=train_data_path,
                data_cache_dir = data_args.data_cache_dir,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                preprocessing_num_workers = data_args.preprocessing_num_workers,
                prompt_strategy = training_args.prompt_strategy,
                use_shots = training_args.use_shots
            )
        logger.info(f"Num train_samples {len(train_dataset)}")
        logger.info("training example ids:")
        logger.info(train_dataset[0]['input_ids'])
        logger.info("label ids:")
        logger.info(train_dataset[0]['labels'])
        labels = train_dataset[0]['labels']
        filtered_labels = [label for label in labels if label != -100]
        logger.info("training example:")
        logger.info(tokenizer.decode(train_dataset[0]['input_ids']))
        logger.info("label:")
        logger.info(tokenizer.decode(filtered_labels))

    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            eval_dataset = build_instruction_dataset(
                data_path=val_data_path,
                data_cache_dir = None,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                preprocessing_num_workers = data_args.preprocessing_num_workers,
                prompt_strategy = training_args.prompt_strategy
            )
        logger.info(f"Num eval_samples {len(eval_dataset)}")
        logger.info("eval example:")
        logger.info(tokenizer.decode(eval_dataset[0]['input_ids']))

    #initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SavePeftModelCallBack]
    )
    
    #training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.continue_finetune)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()



if __name__ == "__main__":
    torch.cuda.empty_cache() 
    main()



