# +
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Dict, Any, Tuple, Mapping
import pdb
import random
import math
import copy
import time



import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.functional import log_softmax

import datasets
from datasets import Dataset
from transformers import is_datasets_available, BatchEncoding
from transformers.trainer import Trainer, logger, skip_first_batches
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    is_torch_tpu_available,
    denumpify_detensorize,
    ShardedDDPOption,
    seed_worker,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    find_batch_size,
    get_parameter_names,
    get_model_param_count,
)
from transformers.optimization import Adafactor, get_scheduler
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.data.data_collator import InputDataClass

from dp_transformers.dp_utils import OpacusDPTrainer

from arguments import TrainingArguments, PrivacyArguments
from metrics import (
    compute_txt_metrics,
    compute_grouped_metrics,
)

from causal_trainer import CausalTrainer


# -

def batch_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features]).squeeze(0)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features])).squeeze(0)
            else:
                batch[k] = torch.tensor([f[k] for f in features]).squeeze(0)

    return batch


class BatchTrainer(CausalTrainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = batch_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))    
