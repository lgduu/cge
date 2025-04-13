# +
# #!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
import json
import random
import pdb
from collections import defaultdict
import copy
import pandas as pd
import datasets
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from tqdm import tqdm
import subprocess

from filelock import FileLock
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    default_data_collator,
    HfArgumentParser,
    BitsAndBytesConfig,
    set_seed,
    LlamaTokenizer,
    LlamaForCausalLM,
    OPTForCausalLM,
    GPT2LMHeadModel,
    FalconForCausalLM
)
from transformers.file_utils import is_offline_mode, is_in_notebook
from transformers.trainer_utils import get_last_checkpoint

from peft import get_peft_model, PeftConfig

from model_distill import LlamaForDistill, FalconForDistill
from trainer_contrast import ContrastTrainer
from causal_collator import DataCollatorForCausalLM
from causal_trainer import CausalTrainer, DPTrainer
from arguments import DataTrainingArguments, ModelArguments, TrainingArguments, PrivacyArguments, PeftArguments, args_to_output_dir

PROXIES = {
    'http': os.environ.get("PROXY"),
    'https': os.environ.get("PROXY"),
}
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "model"
DATA_DIR = "data"

logger = logging.getLogger(__name__)


# -

def configure(argv=None):
    model_args, data_args, training_args, privacy_args, peft_args = parse(argv)
    
    if training_args.contrast or training_args.extract or training_args.distill:
        return _distill(model_args=model_args, data_args=data_args, training_args=training_args, privacy_args=privacy_args, peft_args=peft_args)
    else:
        return _train(model_args=model_args, data_args=data_args, training_args=training_args, privacy_args=privacy_args, peft_args=peft_args)


def parse(argv):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, PrivacyArguments, PeftArguments))

    if argv is not None:
        model_args, data_args, training_args, privacy_args, peft_args = parser.parse_args_into_dataclasses(args=argv.split())
    else:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args, privacy_args, peft_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args, privacy_args, peft_args = parser.parse_args_into_dataclasses()
        argv = " ".join(sys.argv[1:])
        
    # Load pretrained model and tokenizer
    if model_args.model_name_or_path is None:
        if "open_llama" in model_args.model:
            model_args.model_name_or_path = f"openlm-research/{model_args.model}"
        elif "Llama" in model_args.model:
            model_args.model_name_or_path = f"meta-llama/{model_args.model}"
        elif "opt" in model_args.model:
            model_args.model_name_or_path = f"facebook/{model_args.model}"
        elif "gpt2" in model_args.model:
            model_args.model_name_or_path = f"openai-community/{model_args.model}"
        elif "falcon" in model_args.model:
            model_args.model_name_or_path = f"tiiuae/{model_args.model}"
    else:
        model_args.model_name_or_path = model_args.model_name_or_path
       
    if "open_llama" in model_args.model_name_or_path:
        model_args.model_type = "open_llama"
    elif "Llama" in model_args.model_name_or_path or "Swallow" in model_args.model_name_or_path:
        model_args.model_type = "llama"
    elif "opt" in model_args.model_name_or_path:
        model_args.model_type = "opt"
    elif "gpt2" in model_args.model_name_or_path:
        model_args.model_type = "gpt2"
    elif "falcon" in model_args.model_name_or_path:
        model_args.model_type = "falcon"
        
    if training_args.output_dir is None:
        if model_args.model_type == "open_llama":
            model_dir = os.path.join(CUR_DIR, MODEL_DIR, "llama")
        else:
            model_dir = os.path.join(CUR_DIR, MODEL_DIR, model_args.model_type)
        training_args.output_dir = args_to_output_dir(argv, model_dir=model_dir)
    else:
        training_args.output_dir = os.path.join(CUR_DIR, training_args.output_dir)
        
    logger.warning(
        f"Output dir:\n{training_args.output_dir.replace(CUR_DIR+'/', '')}"
    )
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model
    set_seed(training_args.seed)
    
    return model_args, data_args, training_args, privacy_args, peft_args


def load(model_args, data_args, training_args, peft_args):
    if model_args.model_type == "open_llama":
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        ModelForCausalLM = LlamaForCausalLM
        ModelForDistill = LlamaForDistill
    elif model_args.model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        ModelForCausalLM = LlamaForCausalLM
        ModelForDistill = LlamaForDistill
    elif model_args.model_type == "falcon":
        tokenizer = AutoTokenizer.from_pretrained(
            "tiiuae/falcon-rw-1b",
            padding_side = "left",
        ) # tiiuae/falcon-rw-7b does not have bos_token
        tokenizer.pad_token = tokenizer.eos_token
        
        ModelForCausalLM = FalconForCausalLM
        ModelForDistill = FalconForDistill
        
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )   
    if training_args.scratch:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            use_auth_token=True if model_args.use_auth_token else None,
            proxies=PROXIES,
        )
        model = AutoModelForCausalLM.from_config(
            config=config,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_flash_attention_2=model_args.fa2,
        )
    else:
        model = ModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_flash_attention_2=model_args.fa2,
        )
        
    if training_args.peft == "lora":
        model = get_peft_model(model=model, peft_config=peft_args.as_lora_config())
            
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size: model.resize_token_embeddings(len(tokenizer))
        
    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )
    
    if training_args.contrast or training_args.extract:
        model_target = ModelForDistill.from_pretrained(
            model_args.target_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_flash_attention_2=model_args.fa2,
        )
        model_target.init_distill(
            model_distill=model,
            freeze_target=True if training_args.contrast else False,
            freeze_distill=True if training_args.unlearn else False,
        )
    elif training_args.distill:
        model_target = AutoModelForCausalLM.from_pretrained(
            model_args.target_model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            use_flash_attention_2=model_args.fa2,
        )
    else:
        model_target = None
        
    if model_target is not None:
        model_target.generation_config.pad_token_id = tokenizer.pad_token_id

        embedding_size = model_target.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size: model_target.resize_token_embeddings(len(tokenizer))
            
        if (
            hasattr(model_target.config, "max_position_embeddings")
            and model_target.config.max_position_embeddings < data_args.max_source_length
        ):
            if model_args.resize_position_embeddings is None:
                logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model_target.config.max_position_embeddings} "
                    f"to {data_args.max_source_length}."
                )
                model_target.resize_position_embeddings(data_args.max_source_length)
            elif model_args.resize_position_embeddings:
                model_target.resize_position_embeddings(data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model_target.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model_target.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )
        
    return tokenizer, model, model_target


def _train(model_args, data_args, training_args, privacy_args, peft_args):
    tokenizer, model, _ = load(model_args, data_args, training_args, peft_args)
    
    generator = torch.Generator()
    generator.manual_seed(training_args.seed)
    
    if training_args.do_train:
        train_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, train_dir_item)) for train_dir_item in data_args.train_dir]
        if len(train_dataset) == 1: train_dataset = train_dataset[0]
            
        if data_args.max_train_samples is not None:
            sampler = RandomSampler(train_dataset, num_samples=data_args.max_train_samples, generator=generator)
            train_dataset = train_dataset.select(sampler)
            
        train_dataset = train_dataset.remove_columns(["dataset"])
                    
    if training_args.do_eval:
        eval_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, eval_dir_item)) for eval_dir_item in data_args.eval_dir]
        if len(eval_dataset) == 1: eval_dataset = eval_dataset[0]
            
        if data_args.max_eval_samples is not None:
            sampler = RandomSampler(eval_dataset, num_samples=data_args.max_eval_samples, generator=generator)
            eval_dataset = eval_dataset.select(sampler)
            
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model=model,
        data_args=data_args,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        dp=privacy_args.dp,
    )
    
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 
    
    # Initialize our Trainer
    if not privacy_args.dp:
        trainer = CausalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    else:    
        trainer = DPTrainer(
            model=model,    
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            privacy_args=privacy_args,
        )
    return trainer, data_args


def _distill(model_args, data_args, training_args, privacy_args, peft_args):
    tokenizer, model, model_target = load(model_args, data_args, training_args, peft_args)

    generator = torch.Generator()
    generator.manual_seed(training_args.seed)
    
    if data_args.train_dir is not None:
        train_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, train_dir_item)) for train_dir_item in data_args.train_dir]
        if len(train_dataset) == 1: train_dataset = train_dataset[0]
            
        if data_args.max_train_samples is not None:
            sampler = RandomSampler(train_dataset, num_samples=data_args.max_train_samples, generator=generator)
            train_dataset = train_dataset.select(sampler)
            
        train_dataset = train_dataset.remove_columns(["dataset"])
                    
    if training_args.do_eval:
        eval_dataset = [load_from_disk(os.path.join(CUR_DIR, DATA_DIR, eval_dir_item)) for eval_dir_item in data_args.eval_dir]
        if len(eval_dataset) == 1: eval_dataset = eval_dataset[0]

        if data_args.max_eval_samples is not None:
            sampler = RandomSampler(eval_dataset, num_samples=data_args.max_eval_samples, generator=generator)
            eval_dataset = eval_dataset.select(sampler)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model=model,
        data_args=data_args,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # we don't want to remove unused columns because we will prepare each batch during training, 
    # and some of the information will aslo be used in evaluation.
    training_args.remove_unused_columns = False 

    # Metric
    trainer = ContrastTrainer(
        model=model,
        model_target=model_target,
        args=training_args,
        train_dataset=train_dataset if data_args.train_dir is not None else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    return trainer, data_args


def run(trainer, data_args):        
    all_metrics = {"run_name": trainer.args.run_name}
    if trainer.args.no_update:
        trainer.sample_distills()#不动态更新，只采样
    else:
        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(trainer.args.output_dir) and trainer.args.do_train and not trainer.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
            if last_checkpoint is None and len(os.listdir(trainer.args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({trainer.args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and trainer.args.resume_from_checkpoint is None:
                logger.warning(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Training
        if trainer.args.do_train:
            checkpoint = None
            if trainer.args.resume_from_checkpoint is not None:
                checkpoint = trainer.args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            if not trainer.args.no_save: trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            if trainer.train_dataset is not None:
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(trainer.train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(trainer.train_dataset))
            else:
                total_train_batch_size = trainer.args.distill_num * trainer.args.gradient_accumulation_steps
                num_train_samples = trainer.args.max_steps * total_train_batch_size
                metrics["train_samples"] = num_train_samples

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            if not trainer.args.no_save: trainer.save_state()

            all_metrics.update(metrics)

        # Evaluation
        if trainer.args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(
                metric_key_prefix="eval",
            )
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(trainer.eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(trainer.eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            all_metrics.update(metrics)

        # Predit
        if trainer.args.do_predict:
            logger.info("*** Predict ***")

            predict_results = trainer.predict(
                trainer.eval_dataset, metric_key_prefix="predict",
                max_new_tokens = trainer.args.max_new_tokens,
                eos_token_id = trainer.args.eos_token_id,
            )
            return predict_results
        
    return all_metrics


if __name__ == "__main__":
    trainer, data_args = configure()
    run(trainer, data_args)
