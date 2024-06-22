import logging
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import datasets
import torch
import random
from datasets import load_dataset, concatenate_datasets
import transformers
import json
from .utils import LANG_TABLE

random.seed(42)
logger = logging.getLogger('__name__')
IGNORE_INDEX = -100

def extract_fields(input_path, output_path, prompt_strategy, use_shots):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    extracted_data = []
    for ex in data:
        item = ex['translation']
        source_lang = item['pair'].split("-")[0]
        target_lang = item['pair'].split("-")[1]
        if prompt_strategy == "intermediate":
            if use_shots:
                shots_eval_dict = item['shots']
            else:
                shots_eval_dict = {}
            input_field = get_inter_prompt(source_lang, target_lang, item, shots_eval_dict)

        elif prompt_strategy == "plain":
            input_field = get_plain_prompt(source_lang, target_lang, item)
            
        output_field = item[target_lang]
        extracted_data.append({"input_field": input_field, "output_field": output_field})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False)
        

def get_inter_prompt(source_lang, target_lang, ex, shots_eval_dict=None):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    shot_prompt = ""
    medfix = f"\nIntermediate Translation: "
    suffix = f"\nFinal {tgt_fullname} Translation: "
    shots = shots_eval_dict
    if shots is not None and len(shots) > 0:
        prefix = f"###You are a good {src_fullname}-{tgt_fullname} translator. Here are some translation examples whose 'Final {tgt_fullname} Translation' is better than the 'Intermediate Translation'.\n\nExamples:"
        for shot in shots:
            shot_src = shot['source']
            shot_med = shot['medium']
            shot_tgt = shot['target']
            shot_prompt += f"\n{src_fullname} Source: " + shot_src + medfix + shot_med + suffix + shot_tgt + "\n"
            instruction = f"\n###Now I have a new translation pair including '{src_fullname} Source' and 'Intermediate Translation'. Learn from the examples, modify the 'Intermediate Translation' to the final {tgt_fullname} translation:\n"

    else:
        prefix = f"###You are a good {src_fullname}-{tgt_fullname} translator.\n"
        instruction = f"###Now I have a translation pair including '{src_fullname} Source' and 'Intermediate Translation'. Modify the 'Intermediate Translation' to the final {tgt_fullname} translation:\n"

    prompt = prefix + shot_prompt + instruction + f"\n{src_fullname} Source: " + ex[source_lang] + medfix + ex['medium'] + suffix

    return prompt


def get_plain_prompt(source_lang, target_lang, ex):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + ex[source_lang] + suffix
    return prompt


def build_instruction_dataset(data_path, data_cache_dir, tokenizer, max_seq_length, preprocessing_num_workers, prompt_strategy, use_shots):

    def tokenization(examples):
        sources = []
        targets = []
        # prompt = PROMPT_TEMPLATE
        for instruction, output in zip(examples['input_field'], examples['output_field']):
            source = instruction
            target = f"{output}{tokenizer.eos_token}"
            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False) #check tokenizer_config.json   "add_bos_token": true, "add_eos_token": false
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False) 

        all_input_ids = []
        all_labels = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {'input_ids':all_input_ids, 'labels': all_labels}
        return results


    if data_cache_dir is None:
        data_cache_dir = str(os.path.dirname(data_path))
        cache_path = os.path.join(data_cache_dir, os.path.basename(data_path).split('.')[0])
    else:
        cache_path = data_cache_dir

    os.makedirs(cache_path, exist_ok=True)
    logger.info(f"cache_path {cache_path}")
    logging.warning("building datasets..")
    
    try:
        instruction_dataset = datasets.load_from_disk(cache_path)
        logger.info(f'training datasets-{data_path} has been loaded from disk')

    except Exception:
        temp_file_path = "temp.json"
        extract_fields(data_path, temp_file_path, prompt_strategy, use_shots)
        raw_dataset = load_dataset("json", data_files=temp_file_path)
        os.remove(temp_file_path)
        tokenization_func = tokenization
        instruction_dataset = raw_dataset.shuffle().map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["input_field","output_field"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
        instruction_dataset.save_to_disk(cache_path)
    instruction_dataset.set_format('torch')
    instruction_dataset = instruction_dataset['train']
    logger.info(f"Processed dataset successfully saved to {cache_path}")

    return instruction_dataset


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

