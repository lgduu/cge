# +
import string
import re
import json
import sys
import os
import logging
from collections import Counter, defaultdict
import unicodedata
from fuzzywuzzy import fuzz
import pdb

from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import numpy as np
# -


logger = logging.getLogger(__name__)

# +
# # adapted the flowing from Squad v1.1 evaluation, without removing the articles.
# def normalize_answer(s):
#     """Lower text and remove punctuation, and extra whitespace."""

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_punc(lower(s)))
# -

def preprocess_logits_for_metrics(logits, inputs):
    return logits[:, inputs["gen_input_ids"].shape[1]:]


def normalize_answer(input_str: str) -> str:
    return unicodedata.normalize("NFKC", input_str)


# +
def compute_metrics(trainer, dataset, preds, labels, group_metrics=True, save_prefix=None, label_pad_token_id=-100):
    preds = np.ma.array(preds, mask=(preds == label_pad_token_id)).filled(fill_value=trainer.tokenizer.pad_token_id)
    preds = trainer.tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.ma.array(labels, mask=(labels == label_pad_token_id)).filled(fill_value=trainer.tokenizer.pad_token_id)
    labels = trainer.tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = [[label] for label in labels]

#     result = compute_txt_metrics(preds=preds, labels=labels, metrics=dataset["metric"])
    result_per_task = compute_grouped_metrics(preds=preds, labels=labels, groups=dataset["dataset"], compute_metrics=compute_txt_metrics, metrics=dataset["metric"])
    result = {"metric": sum(result_per_task.values())/len(result_per_task.values())}
    result.update(result_per_task)
#     categories = ["_".join(it[0].lower().split()) for it in dataset["Categories"]]
#     result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references, groups=categories, compute_metrics=compute_txt_metrics, metrics=training_args.metric)
#     result.update(result_per_category)

    result = {k: round(v, 4) for k, v in result.items()}
    return result


# -

def compute_txt_metrics(preds, labels, metrics, xlingual=False):
    assert len(preds) == len(labels), f"# of preds {len(preds)} doesn't match # of labels {len(labels)}."
    metric_values = defaultdict(int)
    for pred, label, metric in zip(preds, labels, metrics):
        assert isinstance(label, list)
        if "exact_match" == metric:
            metric_values["exact_match"] += metric_max_over_ground_truths(
                exact_match_score, prediction=pred, ground_truths=label, xlingual=xlingual
            )

        elif "char_f1" == metric:
            metric_values["char_f1"] += metric_max_over_ground_truths(
                char_f1_score, prediction=pred, ground_truths=label, xlingual=xlingual
            )
            
        elif "rouge1" == metric:
            metric_values["rouge1"] += metric_max_over_ground_truths(
                rouge1_score, prediction=pred, ground_truths=label, xlingual=xlingual
            )

        elif "rougeL" == metric:
            metric_values["rougeL"] += metric_max_over_ground_truths(
                rougeL_score, prediction=pred, ground_truths=label, xlingual=xlingual
            )
            
        else:
            raise("invalid metric")

    metric_values = {k: v/len(labels) for k, v in metric_values.items()}
    return metric_values


def exact_match_score(prediction, ground_truth, xlingual=False):
    pred = normalize_answer(prediction).split("\n\n")[0]
    true = normalize_answer(ground_truth)
    score = (pred == true)
    pdb.set_trace()
    return score


def char_f1_score(prediction, ground_truth, xlingual=False):
    pred = normalize_answer(prediction).split("\n\n")[0]
    true = normalize_answer(ground_truth)
    score = fuzz.token_sort_ratio(pred, true) / 100.0
    return score


def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_grouped_metrics(preds, labels, groups, compute_metrics, metrics):
    assert len(preds) == len(labels) == len(groups)

    examples_by_group = {}
    for pred, label, group, metric in zip(preds, labels, groups, metrics):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, label, metric))

    results = {}
    for group, group_examples in examples_by_group.items():
        preds, labels, metrics = zip(*group_examples)
        group_metrics = compute_metrics(preds=preds, labels=labels, metrics=metrics)
        for key, value in group_metrics.items():
            results[f"{key}_{group}"] = value
    return results
