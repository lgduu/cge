# +
import logging
import random
import string
import pdb

import torch
import numpy as np

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, default_data_collator
from arguments import DataTrainingArguments

from itertools import chain
import warnings

logger = logging.getLogger(__name__)


# -

@dataclass
class DataCollatorForCausalLM:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any],
        data_args: DataTrainingArguments,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        text_only: bool=False,
        dp: bool=False,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.text_only = text_only
        self.dp = dp
        
        self.padding="max_length" if data_args.pad_to_max_length else "longest"
        self.max_seq_length=data_args.max_seq_length
        
        self.sep_token = f"###"
        self.sep_token_ids = self.tokenizer.encode(self.sep_token, add_special_tokens=False)

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        if "input_ids" in batch[0]: 
            inputs = default_data_collator(batch, return_tensors=return_tensors)
            if "dataset" in batch[0]: 
                inputs["dataset"] = [instance["dataset"] for instance in batch]
            return inputs
            
        inputs = []
        gen_inputs = []
        gen_labels = []
        datasets = []
        categories = []
        separate = False
        seq2seq = False
        for instance in batch:
            input = instance["inputs_pretokenized"]
            
            if "targets_pretokenized" in instance and instance["targets_pretokenized"] is not None:
                separate = True
                
                target = instance["targets_pretokenized"]
                gen_inputs.append(input)
                gen_labels.append(target)
                
                input = f"{input} {self.sep_token} {target}"
                
            inputs.append(input)
            
            if "dataset" in instance: datasets.append(instance["dataset"])

            if "category" in instance: categories.append(instance["category"])
                
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_seq_length,
            padding=self.padding,
            return_tensors=self.return_tensors, 
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=True,
        )
        label_mask = model_inputs["attention_mask"].bool()
        model_inputs["labels"] = model_inputs["input_ids"].masked_fill(~label_mask, self.label_pad_token_id)
                
        if separate:
            # mask labels corresponding input tokens
            input_ids_list = []
            attention_mask_list = []
            token_type_ids_list = []
            labels_list = []

            for model_input_values in zip(*model_inputs.values()):
                model_input = {key: value for key, value in zip(model_inputs.keys(), model_input_values)}
                input_ids = model_input["input_ids"]
                attention_mask = model_input["attention_mask"]
                labels = model_input["labels"]
                token_type_ids = model_input["token_type_ids"] if "token_type_ids" in model_inputs else None
                
                sep_token_ids_start_idx = None

                for idx in np.where(input_ids == self.sep_token_ids[0])[0]:
                    if (self.sep_token_ids == input_ids[idx : idx + len(self.sep_token_ids)].tolist()):
                        sep_token_ids_start_idx = idx

                if sep_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.sep_token}` in the "
                        f"following instance: {self.tokenizer.decode(input_ids)} "
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    labels = torch.full_like(input_ids, self.label_pad_token_id)
                else:
                    sep_token_ids_end_idx = sep_token_ids_start_idx + len(self.sep_token_ids)

                    input_ids = torch.concat([input_ids[:sep_token_ids_start_idx], input_ids[sep_token_ids_end_idx:]])
                    attention_mask = torch.concat([attention_mask[:sep_token_ids_start_idx], attention_mask[sep_token_ids_end_idx:]])
                    labels = torch.concat([labels[:sep_token_ids_start_idx], labels[sep_token_ids_end_idx:]])
                    labels[:sep_token_ids_start_idx] = self.label_pad_token_id
                    if token_type_ids: token_type_ids = torch.concat([token_type_ids[:sep_token_ids_start_idx], token_type_ids[sep_token_ids_end_idx:]])

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                labels_list.append(labels)
                if token_type_ids: token_type_ids_list.append(token_type_ids)
                    
            # create gen_inputs containing only prompts & gen_labels containing only answers
            # we tokenize them again such that each input & label has the same length
            gen_inputs = self.tokenizer(
                gen_inputs,
                max_length=self.max_seq_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                add_special_tokens=True,
            )
            gen_labels = self.tokenizer(
                gen_labels,
                max_length=self.max_seq_length,
                padding=self.padding,
                return_tensors=self.return_tensors, 
                truncation=True,
                pad_to_multiple_of=self.pad_to_multiple_of,
                add_special_tokens=True,
            )
            
            model_inputs = {
                "input_ids": torch.stack(input_ids_list, 0),
                "attention_mask": torch.stack(attention_mask_list, 0),
                "labels": torch.stack(labels_list, 0),
                "gen_input_ids": gen_inputs["input_ids"],
                "gen_attention_mask": gen_inputs["attention_mask"],
                "gen_labels": gen_labels["input_ids"],
            }
            if len(token_type_ids_list) > 0: model_inputs["token_type_ids"] = torch.stack(token_type_ids_list, 0)
            if "token_type_ids" in gen_inputs: model_inputs["gen_token_type_ids"] = gen_inputs["token_type_ids"]
            
        if len(datasets) > 0: model_inputs["dataset"] = datasets
        if len(categories) > 0: model_inputs["category"] = categories
            
        
        if self.dp and ("position_ids" not in model_inputs):
            input_ids = model_inputs["input_ids"]
            model_inputs["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
            
        return model_inputs






