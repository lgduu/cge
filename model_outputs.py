# +
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import (
    ModelOutput,
)


# -

@dataclass
class ContrastLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_target: torch.FloatTensor = None
    logits_distill: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values_distill: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ContrastDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    scores: torch.FloatTensor = None
    scores_target: torch.FloatTensor = None
    scores_distill: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    logits_target: torch.FloatTensor = None
    logits_distill: torch.FloatTensor = None
    logits_contrast: torch.FloatTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GroupBeamSearchDecoderOnlyOutput(ModelOutput):
    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    group_sequences: torch.LongTensor = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
