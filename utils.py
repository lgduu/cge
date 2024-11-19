import os
import re
import pdb
import json
import pandas as pd
import datasets
import numpy as np
from scipy import stats
from itertools import islice


def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def show_log(output_dir, subset=None, last_ckpt=False):
    if last_ckpt:
        last_ckpt = max([int(dirname.replace("checkpoint-", "")) for dirname in os.listdir(output_dir) if "checkpoint-" in dirname])
        path_state = os.path.join(output_dir, "checkpoint-{}".format(last_ckpt), "trainer_state.json")
    else:
        path_state = os.path.join(output_dir, "trainer_state.json")

    with open(path_state, "r") as f:
        log_history = json.load(f)["log_history"]
        
    df_log_raw = pd.DataFrame(log_history)
    df_log = df_log_raw.dropna(subset="eval_loss").dropna(how='all', axis=1)
    if "loss" in df_log_raw.columns:
        df_log_train = df_log_raw.dropna(subset="loss").dropna(how='all', axis=1)
        df_log = df_log.merge(df_log_train, on=["step", "epoch"], how="outer")
    if "step" in df_log.columns: df_log = df_log.set_index("step")
    if subset is not None: df_log = df_log.dropna(subset=subset)
    return df_log


def load_distills(output_dir, tokenizer=None):
    distill_path = os.path.join(output_dir, "distill_ids.json")

    if os.path.exists(distill_path):
        with open(distill_path, "r", encoding="utf-8") as f:
            step_distills = json.load(f)
        step_distills = {int(k): v for k, v in step_distills.items()}
        
        if tokenizer is not None:
            for step, distill_ids in step_distills.items():
                distill_text = tokenizer.batch_decode(distill_ids)
                print(f"Step {step}")
                print(f"{distill_text}")
        
        return step_distills
    else:
        return None


def show_distills(trainer):
    for step, distill_ids in trainer.step_distills.items():
        distill_text = trainer.tokenizer.batch_decode(distill_ids)
        print(f"Step {step}")
        print(f"{distill_text}")
        
    return trainer.step_distills
