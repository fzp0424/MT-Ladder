import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import argparse
from accelerate.utils import set_seed
import logging
from utils.utils import LANG_TABLE
from utils.build_dataset import get_inter_prompt

logger = logging.getLogger(__name__)



def load_pair_dataset(pair, test_file_path):
    test_raw_data = {}
    src_lang = pair.split("-")[0]
    tgt_lang = pair.split("-")[1]
    test_file = os.path.join(test_file_path)
    test_raw_data[f"{src_lang}-{tgt_lang}"] = load_dataset(
        "json",
        data_files={"test": test_file}
        )
    return test_raw_data

def get_pair_suffix(tgt_lang):
    
    return f"\nFinal {LANG_TABLE[tgt_lang]} Translation: "

def get_plain_suffix(tgt_lang):
    
    return f"\n{LANG_TABLE[tgt_lang]}:"

def clean_outputstring(output, key_word, logger, split_idx):
    try:
        out = output.split(key_word)[split_idx].split("\n")
        if out[0].strip() != "":
            return out[0].strip()
        elif out[1].strip() != "":
            ## If there is an EOL directly after the suffix, ignore it
            logger.info(f"Detect empty output, we ignore it and move to next EOL: {out[1].strip()}")
            return "-------------------"
        else:
            logger.info(f"Detect empty output AGAIN, we ignore it and move to next EOL: {out[2].strip()}")
            return "-------------------"
    except:
        logger.info(f"Can not recover the translation by moving to the next EOL.. Trying move to the next suffix")


def get_plain_prompt(source_lang, target_lang, ex, shots_eval_dict):
    src_fullname = LANG_TABLE[source_lang]
    tgt_fullname = LANG_TABLE[target_lang]
    prefix = f"Translate this from {src_fullname} to {tgt_fullname}:\n{src_fullname}: "
    suffix = f"\n{tgt_fullname}:"
    prompt = prefix + ex[source_lang] + suffix
    return prompt

# Function to generate model outputs based on the input data
def generate_model_outputs(row, model):

    input_ids = torch.tensor(row['input_ids']).unsqueeze(dim=0).to('cuda')
    with torch.no_grad():
        generate_ids = model.generate(input_ids=input_ids, num_beams=5, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9)
    model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return model_output

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default=None, help="The name of model to use.");
    parser.add_argument("--peft-path", type=str, default=None, help="The name of model to use.");
    parser.add_argument("--prompt-strategy", type=str, default="intermediate", help="intermediate, plain..");
    parser.add_argument("--test-pairs", type=str, default="", help="en-zh,de-en... no space");
    parser.add_argument("--test-dir", type=str, default="./pair_corpus");
    parser.add_argument("--test-file-path", type=str, default="./pair_corpus");
    parser.add_argument("--output-dir", type=str, default=None);
    parser.add_argument("--output-file-prefix", type=str, default="test");
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    set_seed(args.seed)   
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(args.output_dir)
        
    # test_pairs = args.test_pairs.split(",")
    test_pairs = args.test_pairs
    print(test_pairs)
    test_raw_data = load_pair_dataset(test_pairs, args.test_file_path)
    source_lang = test_pairs.split("-")[0]
    target_lang = test_pairs.split("-")[1]


    # Load base model and LoRA weights
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype='auto', device_map = 'auto')
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path, torch_dtype='auto', device_map = 'auto')
        model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')

    def tokenize_function_test(examples):
        prompts = []
        targets = []
        for ex in examples['translation']:
            if args.prompt_strategy == 'intermediate':
                shots_eval_dict = ex['shots']
                prompt = get_inter_prompt(source_lang, target_lang, ex, shots_eval_dict)
            elif args.prompt_strategy == 'plain':
                shots_eval_dict = {}
                prompt = get_plain_prompt(source_lang, target_lang, ex, shots_eval_dict)
            prompts.append(prompt)
            targets.append(prompt + ex[target_lang])

        original_padding_side = tokenizer.padding_side
        if original_padding_side != "left":
            tokenizer.padding_side = "left"
        model_inputs = tokenizer(prompts, max_length=1024, padding=True, truncation=True, add_special_tokens=True)

        return model_inputs

    test_datasets = {}
    for lg_pair, sub_raw_data in test_raw_data.items():
        test_dataset = sub_raw_data["test"]
        test_dataset = test_dataset.map(
            tokenize_function_test,
            batched=True,
            num_proc=8,
            remove_columns=["translation"],
            desc=f"Running tokenizer {lg_pair} test dataset",
        )
        test_datasets[lg_pair] = test_dataset
        
        input_ids = test_datasets[lg_pair][0]['input_ids']
        decode_check = tokenizer.decode(input_ids, skip_special_tokens=True)
        print("-"*50 + "check input" + "-"*50)
        print(decode_check)
        print("-"*50 + "check input" + "-"*50)


        for idx in tqdm(range(len(test_dataset)), desc="Generating Responses"):
            row = test_dataset[idx]
            output = generate_model_outputs(row, model)
            
            if idx == 0:
                print("-" * 50 + "output example" + "-" * 50)
                print(output)
                print("-" * 50 + "output example" + "-" * 50)
                
            with open(os.path.join(args.output_dir, f"{args.output_file_prefix}-{lg_pair}"), "a", encoding="utf-8") as f:
                target_lan = lg_pair.split("-")[1]
                if args.prompt_strategy == 'intermediate': 
                    suffix = get_pair_suffix(target_lan) 
                elif args.prompt_strategy == 'plain': 
                    suffix = get_plain_suffix(target_lan)
                suffix_count = output.count(suffix)
                
                split_idx = suffix_count
                
                pred = clean_outputstring(output, suffix, logger, split_idx)
                try:
                    f.writelines([pred, "\n"])
                except:
                    f.writelines(["None", "\n"])

