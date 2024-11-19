# +
import pdb
import os
import json

import numpy as np
import pandas as pd
import transformers
import torch
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from torch.distributions import Categorical
from datasets import load_from_disk
from tqdm import tqdm


# -

def compute_probs(model, tokenizer, prompt, text, categories, max_length=None):
    messages = [
        {"role": "system",
         "content": prompt,
        },
        {"role": "user", 
         "content": text},
    ]
    prompts = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        max_length=max_length,
        truncation=True if max_length else False,
    )

    answers = tokenizer(
        list(categories),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )

    inputs = {
        key: torch.concat([prompts[key].expand(len(categories), -1), answers[key]], axis=1).to(model.device)
        for key in prompts.keys()
    }
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        mask_labels = inputs["attention_mask"].bool()
        mask_labels[:, :prompts["input_ids"].size(-1)] = False
        labels = inputs["input_ids"]
        labels = labels.masked_fill(~mask_labels, -100)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        losses_flat = loss_fct(shift_logits, shift_labels) # batch_size*seq_length
        losses_seq = losses_flat.view(logits.size(0), -1) # batch_size x seq_length
        losses = losses_seq.sum(-1) / mask_labels.sum(-1) # batch_size
        probs = torch.softmax(-losses, -1).cpu()
        
    sorted_probs = probs.sort(descending=True)
    index = sorted_probs.indices[0]
    pred = categories[index]
    prob = sorted_probs.values[0].item()

    return pred, prob


def compute_ppl(model, tokenizer, text, max_length=None):
    inputs = tokenizer(
        text, 
        max_length=max_length,
        truncation=True if max_length else False,
        return_tensors="pt",
    ).to(model.device)
    inputs["labels"] = inputs.input_ids
    
    with torch.no_grad():
        outputs = model(**inputs)
    ppl = torch.exp(outputs.loss).item()
    return ppl


def compute_ppls(model, tokenizer, texts, max_length=None):
#     all_distill_texts = torch.tensor(list(step_distills.values())).squeeze(1)
    texts_loader = batched(texts, n=8)
    for texts in texts_loader:
        inputs = tokenizer(
            texts, 
            max_length=max_length,
            truncation=True if max_length else False,
            return_tensors="pt",
        ).to(model.device)
        inputs["labels"] = inputs.input_ids

        with torch.no_grad():
            outputs = model(**inputs)

        break


def generate_preds(model, tokenizer, prompt, text):
    messages = [
        {"role": "system",
         "content": prompt,
        },
        {"role": "user", 
         "content": text},
    ]
    prompts = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model.generate(**prompts)
        
    return outputs


def compute_scores(trainer, gamma, norm=True):
    scores_all = []
    datasets_all = []
    categories_all = []
    texts_all = []

    trainer.model_target.gamma = gamma
    
    dataloader = trainer.get_eval_dataloader()
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            model_inputs = trainer._prepare_inputs(inputs)
            if norm:
                losses = trainer.compute_losses_distill(trainer.model_target, model_inputs)
            else:
                losses = trainer.compute_unnorm_losses_distill(trainer.model_target, model_inputs)
            scores = losses.cpu().to(torch.float32).tolist()
            datasets = inputs["dataset"]
            categories = inputs["category"].tolist()
            texts = trainer.tokenizer.batch_decode(inputs["input_ids"])

            scores_all += scores
            datasets_all += datasets
            categories_all += categories
            texts_all += texts

    scores_df = pd.DataFrame({
        "text": texts_all,
        "datasets": datasets_all,
        "labels": categories_all,
        "scores": scores_all,
    })
    scores_df = scores_df.sort_values("scores", ascending=False).reset_index()

    return scores_df


def compute_prob_scores(trainer, msp=False, energy=False, entropy=False, gen=False, gamma=None):
    scores_all = []
    datasets_all = []
    categories_all = []
    texts_all = []

    dataloader = trainer.get_eval_dataloader()
    for inputs in tqdm(dataloader):
        with torch.no_grad():
            model_inputs = trainer._prepare_inputs(inputs)
            if msp:
                outputs = trainer.model(**model_inputs)
                logits = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
                shift_logits = logits[..., :-1, :].contiguous()
                
                shift_labels = model_inputs["labels"][..., 1:].contiguous()
                mask_labels = shift_labels != trainer.tokenizer.pad_token_id
                losses_seq = shift_logits.max(-1).values # max logits
                losses = torch.sum(losses_seq * mask_labels, -1) / mask_labels.sum(-1)
            elif energy:
                outputs = trainer.model(**model_inputs)
                energies = -outputs.logits.sum(-1) # sum over tokens
                losses = -energies.mean(-1) # average over sequence
            elif entropy:
                outputs = trainer.model(**model_inputs)
                entropies = Categorical(logits=outputs.logits).entropy()
                losses = -entropies.mean(-1) # average over sequence
            elif gen:
                outputs = trainer.model(**model_inputs)
                probs = torch.softmax(outputs.logits, -1)
                gen_entropies = ((probs*(1-probs))**gamma).sum(-1)
                losses = -gen_entropies.mean(-1)
            else:
                losses = -trainer.compute_losses(trainer.model, model_inputs)
            scores = (losses).cpu().to(torch.float32).tolist()
            datasets = inputs["dataset"]
            categories = inputs["category"].tolist()
            texts = trainer.tokenizer.batch_decode(inputs["input_ids"])

            scores_all += scores
            datasets_all += datasets
            categories_all += categories
            texts_all += texts

    scores_df = pd.DataFrame({
        "text": texts_all,
        "datasets": datasets_all,
        "labels": categories_all,
        "scores": scores_all,
    })
    scores_df = scores_df.sort_values("scores", ascending=False).reset_index()
    return scores_df


def compute_grad_scores(trainer, uniform=False):
    scores_all = []
    datasets_all = []
    categories_all = []
    texts_all = []

    norm = lambda grad: sum([torch.sum(g**2) for g in grad]).item()
    dataloader = trainer.get_eval_dataloader()
    for inputs in tqdm(dataloader):
        model_inputs = trainer._prepare_inputs(inputs)
        if uniform:
            losses = trainer.compute_uniform_losses(trainer.model, model_inputs)
        else:
            losses = trainer.compute_losses(trainer.model, model_inputs)
        scores = [-norm(torch.autograd.grad(loss, trainer.model.parameters(), retain_graph=True)) for loss in losses]
        datasets = inputs["dataset"]
        categories = inputs["category"].tolist()
        texts = trainer.tokenizer.batch_decode(inputs["input_ids"])

        scores_all += scores
        datasets_all += datasets
        categories_all += categories
        texts_all += texts

    scores_df = pd.DataFrame({
        "text": texts_all,
        "datasets": datasets_all,
        "labels": categories_all,
        "scores": scores_all,
    })
    scores_df = scores_df.sort_values("scores", ascending=False).reset_index()
    return scores_df


def compute_knn_scores(trainer, k=1):
    train_dataloader = trainer.get_train_dataloader()
    eval_dataloader = trainer.get_eval_dataloader()
    
    if trainer.model.config.model_type == "llama":
        model = trainer.model.model
    elif trainer.model.config.model_type == "falcon":
        model = trainer.model.transformer
    else:
        raise("not implemented")

    train_outputs = None
    for inputs in tqdm(train_dataloader):
        inputs = trainer._prepare_inputs(inputs)
        inputs.pop("labels")

        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1)
        outputs = torch.nn.functional.normalize(outputs)

        train_outputs = outputs if train_outputs is None else torch.concat([train_outputs, outputs], dim=0)

    knn_texts = None
    knn_categories = None
    knn_dists = None
    for eval_inputs in tqdm(eval_dataloader):
        eval_categories = eval_inputs["category"]
        eval_texts = trainer.tokenizer.batch_decode(eval_inputs["input_ids"])
        eval_inputs = trainer._prepare_inputs(eval_inputs)
        eval_inputs.pop("labels")

        with torch.no_grad():
            eval_outputs = model(**eval_inputs).last_hidden_state.mean(dim=1)
        eval_outputs = torch.nn.functional.normalize(eval_outputs)

        eval_dists = None
        diff_outputs = eval_outputs[:, None, :] - train_outputs[None, :, :]
        diff_dists = torch.norm(diff_outputs, dim=-1)

        eval_dists = diff_dists if eval_dists is None else torch.concat([eval_dists, diff_dists], dim=-1)

        knn_texts = eval_texts if knn_texts is None else knn_texts + eval_texts
        knn_categories = eval_categories if knn_categories is None else torch.concat([knn_categories, eval_categories], dim=0)
        knn_dists = eval_dists if knn_dists is None else torch.concat([knn_dists, eval_dists], dim=0)
        
    knn_scores_df = {}
    knn_scores_df["texts"] = knn_texts
    knn_scores_df["labels"] = knn_categories
    knn_scores_df["dists"] = knn_dists.tolist()

    knn_scores_df = pd.DataFrame(knn_scores_df)
    knn_scores_df["scores"] = (-torch.topk(torch.tensor(knn_scores_df["dists"]), k=k, largest=False, dim=-1).values[:, -1]).tolist()
    return knn_scores_df
