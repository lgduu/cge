# +
import os
import pdb
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List, Tuple
from collections import Counter
import re

from transformers import Seq2SeqTrainingArguments, logging
from transformers.training_args import OptimizerNames, trainer_log_levels
from transformers.utils import ExplicitEnum
from transformers.trainer_utils import EvaluationStrategy, HubStrategy, IntervalStrategy, SchedulerType

from dp_transformers.arguments import find_noise_multiplier
from peft import get_peft_model, LoraConfig

logger = logging.get_logger(__name__)


# -

def args_to_output_dir(argv, model_dir=None):
    abbr_dict = {
        "train_adapter": "adapter",
        "per_device_train_batch_size": "train_batch",
        "gradient_accumulation_steps": "accumul",
        "train_dir": "train",
        "retain_dir": "retain",
        "learning_rate": "lr",
        "warmup_steps": "warmup",
        "lr_scheduler_type": "schedule",
        "noise_multiplier": "noise",
        "distill_top_p": "top_p",
        "distill_do_sample": "sample",
        "distill_num_beams": "beam",
        "distill_num_beam_groups": "group",
        "distill_diversity_penalty": "diversity",
        }
    del_list = [
        "torch_dtype",
        "do_train",
        "do_eval",
        "do_predict",
        "fp16_full_eval",
        "bf16_full_eval",
        "generation_max_length",
        "generation_max_new_tokens",
        "per_device_eval_batch_size",
        "log_level",
        "logging_steps",
        "logging_strategy",
        "no_save",
        "save_steps",
        "save_strategy",
        "eval_dir",
        "eval_steps",
        "evaluation_strategy",
        "max_steps",
        "num_train_epochs",
        "predict_with_generate",
        "save_arnoldi",
        "overwrite_output_dir",
    ]
    
    args = argv.strip().split("--")[1:]
    args = [arg.split("/")[-1] if "/" in arg else arg for arg in args] # model_name_or_path removed
    args = [arg for arg in args if not re.match("|".join(del_list), arg)]
    output_dir = "-".join([arg.strip().replace(" ", "=").replace("model=", "") for arg in args])
    for arg, abbr in abbr_dict.items():
        output_dir = output_dir.replace(arg, abbr)
    if model_dir: output_dir = os.path.join(model_dir, output_dir)
    return output_dir


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # super arguments
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    distill_path: str = field(
        default="distill_ids.json",
    )
    candidate_path: str = field(
        default="candidate_ids.pt",
    )
    candidate_score_path: str = field(
        default="candidate_scores.pt",
    )
    candidate_dir: str = field(
        default=None,
    )
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
        
    generation_max_new_tokens: Optional[int] = field(
        default=16,
    )

    contrast: bool = field(
        default=False, 
    )
    extract: bool = field(
        default=False, 
    )
    distill: bool = field(
        default=False, 
    )
    unnorm: bool = field(
        default=False, 
    )
    baseline: bool = field(
        default=False, 
    )
        
    unlearn: bool = field(
        default=False, 
    )
    peft: str = field(
        default=None,
    )
    legacy: bool = field(
        default=False, 
    )
    batch: bool = field(
        default=False, 
    )
        
    distill_length: int = field(
        default=256,
    )
    distill_new_tokens: int = field(
        default=None,
    )
    distill_num: int = field(
        default=1,
    )
    distill_epochs: int = field(
        default=None,
    )
    distill_steps: int = field(
        default=None,
    )
    distill_do_sample: bool = field(
        default=True, 
    )
    distill_num_beams: int = field(
        default=1, 
    )
    distill_num_beam_groups: int = field(
        default=1, 
    )
    distill_diversity_penalty: float = field(
        default=0., 
    )
    distill_top_k: int = field(
        default=None, 
    )
    distill_top_p: float = field(
        default=None, 
    )
    distill_typical_p: float = field(
        default=None, 
    )
    candidate_num: int = field(
        default=None,
    )

    no_update: bool = field(
        default=False, 
    )
    alpha: float = field(
        default=0., 
    )
    beta: float = field(
        default=0., 
    )
    kappa: int = field(
        default=0, 
    )
    gamma: Optional[float] = field(
        default=1.,
    )
    target_top_p: float = field(
        default=1.0, 
    )
    target_typical_p: float = field(
        default=1.0, 
    )
    use_past_distills: bool = field(
        default=False, 
    )
    accumulate: bool = field(
        default=False, 
    )
        
    rmsprop: bool = field(
        default=False, 
    )
    rms_alpha: float = field(
        default=None, 
    )
    rms_eps: float = field(
        default=None, 
    )
        
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
        
    scratch: bool = field(
        default=False, 
    )
    optim: OptimizerNames = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: SchedulerType = field(
        default="constant_with_warmup",
        metadata={"help": "The scheduler type to use."},
    )
    momentum: float = field(
        default=0.
    )    
    label_names: Optional[List[str]] = field(
        default_factory=lambda: ['labels'], metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
        
    logging_steps: int = field(default=10000, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=10000, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints. "
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_save: bool = field(
        default=False, 
    )
    eval_steps: int = field(default=10000, metadata={"help": "Run an evaluation every X steps."})
    evaluation_strategy: IntervalStrategy = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    log_level: Optional[str] = field(
        default="warning",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
                
    parallelize: bool = field(
        default=False,
    )

    debug_grad_loss: bool = field(
        default=False,
    )
    debug_grad_model: bool = field(
        default=False,
    )
    tmp: bool = field(
        default=False, 
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    checkpoint: str = field(
        default=None,
    )
    causal: bool = field(
        default=False, 
    )
    model: Optional[str] = field(
        default=None, 
    )
    model_name_or_path: Optional[str] = field(
        default=None, 
    )
    target_model_name_or_path: Optional[str] = field(
        default=None, 
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    fa2: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attention 2."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
        
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total sequence length after tokenization (used for decoder only model). Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
        
    train_dir: List[str] = field(
        default=None,
    )
    eval_dir: List[str] = field(
        default=None,
    )
        
    dataset_names: Optional[List[str]] = field(
        default=None,
    )
    loo_dataset_names: Optional[List[str]] = field(
        default=None,
    )
        
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
            
    def __post_init__(self):
        pass


@dataclass
class PrivacyArguments:
    per_sample_max_grad_norm: Optional[float] = field(default=1., metadata={"help": "Max per sample clip norm"})
    noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    target_epsilon: Optional[float] = field(default=None, metadata={
        "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    })
    target_delta: Optional[float] = field(default=None, metadata={
        "help": "Target delta, defaults to 1/N"
    })
#     disable_dp: bool = field(default=True, metadata={
#         "help": "Disable DP training."
#     })
    dp: bool = field(default=False, metadata={
        "help": "Turn on DP training."
    })

        
    def initialize(self, sampling_probability: float, num_steps: int, num_samples: int) -> None:
        if self.target_delta is None:
            self.target_delta = 1.0/num_samples
        logger.info(f"The target delta is set to be: {self.target_delta}")

        # Set up noise multiplier
        if self.noise_multiplier is None:
            self.noise_multiplier = find_noise_multiplier(
                sampling_probability=sampling_probability,
                num_steps=num_steps,
                target_delta=self.target_delta,
                target_epsilon=self.target_epsilon
            )
        logger.info(f"The noise multiplier is set to be: {self.noise_multiplier}")

    @property
    def is_initialized(self) -> bool:
        return (
            self.per_sample_max_grad_norm is not None and
            self.noise_multiplier is not None and
            self.target_delta is not None
        )

    def __post_init__(self):
        if not self.dp:
            logger.warning("Disabling differentially private training...")
            self.noise_multiplier = 0.0
            self.per_sample_max_grad_norm = float('inf')
            self.target_epsilon = None
        else:
            if bool(self.target_epsilon) == bool(self.noise_multiplier):
                raise ValueError("Exactly one of the arguments --target_epsilon and --noise_multiplier must be used.")
            if self.per_sample_max_grad_norm is None:
                raise ValueError("DP training requires --per_sample_max_grad_norm argument.")


@dataclass
class PeftArguments:
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=16, metadata={
        "help": "LoRA alpha"
    })
    target_modules: Optional[List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",], 
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    lora_dropout: float = field(default=0.05, metadata={
        "help": "LoRA dropout"
    })

    def as_lora_config(self) -> LoraConfig:
        params = asdict(self)
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)
