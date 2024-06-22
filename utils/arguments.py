from dataclasses import dataclass, field
from typing import Optional
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments
from transformers.utils.versions import require_version
from trl import ORPOConfig

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
@dataclass
class ModelArguments:
    """
    Arguments pretaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":(
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        }
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization. Don't set if you want to train a model from scratch."
            )
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help":"Where do you want to store the pretrained models downloaded from hf"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizer library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help":(
                "Will use the token generated when running `huggingface-cli login`(necessary to use this script"
                "with private models)."
            )
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help":(
                "Override the default 'torch.dtype' and load the model under this dtype. If 'auto' is passed, the"
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16","float32"],
        }
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertraining to what data we are going to input our model for training and eval.
    """

    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}   
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optinal input evaluation data file to evaluate the perplexity on (a text file)."}
    )

    overwrite_cache: bool = field(
        default=False, metadata={"help":"Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    data_cache_dir: Optional[str] = field(default=None, metadata={"help":"The datasets processed stored."})

    max_seq_length: Optional[int] = field(default=512)

@dataclass
class MyTrainingArguments(TrainingArguments):

    trainable: Optional[str] = field(default="q_proj,v_proj")
    use_dora: bool = field(default=False)
    use_adalora: bool = field(default=False)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    init_rank: Optional[int] = field(default=12)
    target_rank: Optional[int] = field(default=8)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    use_wandb: bool = field(default=True)
    wandb_project: Optional[str] = field(default=None)
    wandb_watch: Optional[str] = field(default=None)
    wandb_log_model: Optional[str] = field(default=None)
    continue_finetune: bool = field(default=False)
    prompt_strategy: Optional[str] = field(default="plain")
    use_shots: bool = field(default=True)
    dpo_beta: float = field(default=0.1)
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    orpo_beta: float = field(default=0.1)
    model_init_kwargs: Optional[str] = field(default=None,      
        metadata={
            "help": (
                "For ORPO"
            )
        },)
    generate_during_eval: Optional[str] = field(default=None,
        metadata={
            "help": (
                "For ORPO"
            )
        },)
    

@dataclass
class ORPOTrainingArguments(ORPOConfig):

    trainable: Optional[str] = field(default="q_proj,v_proj")
    use_dora: bool = field(default=False)
    use_adalora: bool = field(default=False)
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_alpha: Optional[float] = field(default=32.)
    init_rank: Optional[int] = field(default=12)
    target_rank: Optional[int] = field(default=8)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default=None)
    use_wandb: bool = field(default=True)
    wandb_project: Optional[str] = field(default=None)
    wandb_watch: Optional[str] = field(default=None)
    wandb_log_model: Optional[str] = field(default=None)
    continue_finetune: bool = field(default=False)
    prompt_strategy: Optional[str] = field(default="plain")
    use_shots: bool = field(default=True)
    dpo_beta: float = field(default=0.1)
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum total sequence length text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "The maximum new tokens to generate except the prompt."
            )
        },
    )
    
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    max_prompt_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "For ORPO"
            )
        },
    )
    orpo_beta: float = field(default=0.1)

