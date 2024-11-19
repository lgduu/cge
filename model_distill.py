# +
import pdb
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.functional import log_softmax

from transformers import AutoModelForCausalLM, LlamaForCausalLM, OPTForCausalLM, GPT2LMHeadModel, FalconForCausalLM, PreTrainedModel
from transformers.modeling_outputs import (
    ModelOutput,
)
from transformers import BeamScorer

from transformers.generation import LogitsWarper, StoppingCriteriaList, GenerationMixin
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin
from transformers.generation.utils import SampleOutput, BeamSearchOutput, BeamSearchDecoderOnlyOutput, BeamSampleOutput, BeamSampleDecoderOnlyOutput
from peft import PeftConfig, PeftModelForCausalLM

from model_outputs import ContrastLMOutputWithPast, ContrastDecoderOnlyOutput, GroupBeamSearchDecoderOnlyOutput


# +
class BaseModelForDistill(nn.Module):
    def init_distill(
        self,
        model_distill,
        freeze_target=False,
        freeze_distill=False,
    ):
        if freeze_target:
            # freeze model_target now such that model_distill is not frozen
            for param in self.parameters():
                param.requires_grad_(False)

        if freeze_distill:
            for param in model_distill.parameters():
                param.requires_grad_(False)
        self.model_distill = model_distill

        self.config.eos_token_id = None
        self.generation_config.eos_token_id = None
        self.eval()
    
    def forward_distill(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_distill: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ContrastLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
        if self.config.model_type == "llama":
            outputs_target = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif self.config.model_type == "opt":
            outputs_target = self.model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif self.config.model_type == "gpt2":
            outputs_target = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.config.model_type == "falcon":
            outputs_target = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
        outputs_target.logits = self.lm_head(outputs_target[0]).contiguous()
        logits_target = outputs_target.logits.float()
        logits_target = log_softmax(logits_target, -1)
                    
        if self.gamma != 0:
            outputs_distill = self.model_distill(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_distill,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            logits_distill = outputs_distill.logits.float()
            logits_distill = log_softmax(logits_distill, -1)

            logits = logits_target - self.gamma * logits_distill
        else:
            outputs_distill = None
            logits_distill = None
            logits = logits_target

        loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return ContrastLMOutputWithPast(
            loss=loss,
            logits=logits,
            logits_target=logits_target,
            logits_distill=logits_distill,
            past_key_values=outputs_target.past_key_values,
            past_key_values_distill=outputs_distill.past_key_values if outputs_distill is not None else None,
            hidden_states=None,
            attentions=None,
        )
    
    def get_logits_warper(self):
        target_filter_value = -float("Inf")
        if self.alpha > 0:
            self.target_logits_warper = PlausibleLogitsWarper(alpha=self.alpha, filter_value=target_filter_value)
        else:
            self.target_logits_warper = None
            
        distill_filter_value = float("Inf")
        if self.beta > 0:
            self.distill_logits_warper = MinPLogitsWarper(beta=self.beta)
#             self.distill_logits_warper = TopPLogitsWarper(top_p=self.beta, filter_value=distill_filter_value)
        elif self.kappa > 0:
            self.distill_logits_warper = MinKLogitsWarper(kappa=self.kappa)
        else:
            self.distill_logits_warper = None

        return self.target_logits_warper, self.distill_logits_warper
    
    def greedy_search(
        self,
        *args,
        **kwargs,
    ):
        raise("not implemented")
            
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        scores_target = () if (return_dict_in_generate and output_scores) else None
        scores_distill = () if (return_dict_in_generate and output_scores) else None
        logits = () if (return_dict_in_generate and output_scores) else None
        logits_target = () if (return_dict_in_generate and output_scores) else None
        logits_distill = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
            
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.forward_distill(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # pre-process distribution
            next_token_logits = outputs.logits[:, -1, :]
            
            next_token_logits_target = outputs.logits_target[:, -1, :]
            if self.target_logits_warper is not None:
                next_token_scores_target = self.target_logits_warper(None, next_token_logits_target)
            else:
                next_token_scores_target = next_token_logits_target
            
            if outputs.logits_distill is not None:
                next_token_logits_distill = outputs.logits_distill[:, -1, :]
                if self.distill_logits_warper is not None:
                    next_token_scores_distill = self.distill_logits_warper(None, next_token_logits_distill)
                else:
                    next_token_scores_distill = next_token_logits_distill
                
                next_token_scores = next_token_scores_target - self.gamma * next_token_scores_distill
            else:
                next_token_scores = next_token_scores_target
            
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                    logits += (next_token_logits,)
                    scores_target += (next_token_scores_target,)
                    logits_target += (next_token_logits_target,)
                    if outputs.logits_distill is not None:
                        scores_distill += (next_token_scores_distill,)
                        logits_distill += (next_token_logits_distill,)
                        
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
                
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return ContrastDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=torch.stack(scores, 1),
                    scores_target=torch.stack(scores_target, 1),
                    scores_distill=torch.stack(scores_distill, 1) if len(scores_distill) > 0 else None,
                    logits=torch.stack(logits, 1),
                    logits_target=torch.stack(logits_target, 1),
                    logits_distill=torch.stack(logits_distill, 1) if len(logits_distill) > 0 else None,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.forward_distill(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            
#             next_token_scores = nn.functional.log_softmax(
#                 next_token_logits, dim=-1
#             )  # (batch_size * num_beams, vocab_size)

            # add
            next_token_logits_target = outputs.logits_target[:, -1, :]
            if self.target_logits_warper is not None:
                next_token_scores_target = self.target_logits_warper(None, next_token_logits_target)
            else:
                next_token_scores_target = next_token_logits_target
            
            if outputs.logits_distill is not None:
                next_token_logits_distill = outputs.logits_distill[:, -1, :]
                if self.distill_logits_warper is not None:
                    next_token_scores_distill = self.distill_logits_warper(None, next_token_logits_distill)
                else:
                    next_token_scores_distill = next_token_logits_distill
                
                next_token_scores = next_token_scores_target - self.gamma * next_token_scores_distill
            else:
                next_token_scores = next_token_scores_target
            # add

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                    
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=torch.stack(scores, 1),
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=torch.stack(scores, 1),
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
        

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.forward_distill(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

#             next_token_scores = nn.functional.log_softmax(
#                 next_token_logits, dim=-1
#             )  # (batch_size * num_beams, vocab_size)

            # add
            next_token_logits_target = outputs.logits_target[:, -1, :]
            if self.target_logits_warper is not None:
                next_token_scores_target = self.target_logits_warper(None, next_token_logits_target)
            else:
                next_token_scores_target = next_token_logits_target
            
            if outputs.logits_distill is not None:
                next_token_logits_distill = outputs.logits_distill[:, -1, :]
                if self.distill_logits_warper is not None:
                    next_token_scores_distill = self.distill_logits_warper(None, next_token_logits_distill)
                else:
                    next_token_scores_distill = next_token_logits_distill
                
                next_token_scores = next_token_scores_target - self.gamma * next_token_scores_distill
            else:
                next_token_scores = next_token_scores_target
            # add

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )
            # Note: logits warpers are intentionally applied after adding running beam scores. On some logits warpers
            # (like top_p) this is indiferent, but on others (like temperature) it is not. For reference, see
            # https://github.com/huggingface/transformers/pull/5420#discussion_r449779867
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logits_warper(input_ids, next_token_scores_processed),)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
        
    def group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape

        if return_dict_in_generate and output_scores:
            beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
        else:
            beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.forward_distill(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if output_scores:
                processed_score = torch.zeros_like(outputs.logits[:, -1, :])

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]
                
#                 next_token_scores = nn.functional.log_softmax(
#                     next_token_logits, dim=-1
#                 )  # (batch_size * num_beams, vocab_size)
                
                # add
                next_token_logits_target = outputs.logits_target[batch_group_indices, -1, :]
                if self.target_logits_warper is not None:
                    next_token_scores_target = self.target_logits_warper(None, next_token_logits_target)
                else:
                    next_token_scores_target = next_token_logits_target

                if outputs.logits_distill is not None:
                    next_token_logits_distill = outputs.logits_distill[batch_group_indices, -1, :]
                    if self.distill_logits_warper is not None:
                        next_token_scores_distill = self.distill_logits_warper(None, next_token_logits_distill)
                    else:
                        next_token_scores_distill = next_token_logits_distill

                    next_token_scores = next_token_scores_target - self.gamma * next_token_scores_distill
                else:
                    next_token_scores = next_token_scores_target
                # add
                
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                if output_scores:
                    processed_score[batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
                n_eos_tokens = len(eos_token_id) if eos_token_id else 0
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, max(2, 1 + n_eos_tokens) * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                if return_dict_in_generate and output_scores:
                    beam_indices[beam_group_idx] = tuple(
                        beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                    )

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self._reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        
        group_input_ids = input_ids.view(num_beam_groups, num_sub_beams, -1)
        group_sequences = group_input_ids[:, 0, :]

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GroupBeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    group_sequences=group_sequences,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


# -

class PeftModelForDistill(PeftModelForCausalLM, BaseModelForDistill):
    def __init__(
        self, 
        model: torch.nn.Module, 
        peft_config: PeftConfig, 
        adapter_name: str = "default", 
        **kwargs,
    ):
        super().__init__(model, peft_config, adapter_name, **kwargs)


class LlamaForDistill(LlamaForCausalLM, BaseModelForDistill):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        
        model_kwargs = LlamaForCausalLM._update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, standardize_cache_format)
        
        # update past_key_values
        past_key_values_distill = outputs.past_key_values_distill
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values_distill = self._convert_to_standard_cache(past_key_values_distill, batch_size=batch_size)
        model_kwargs["past_key_values_distill"] = past_key_values_distill
        return model_kwargs
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = LlamaForCausalLM.prepare_inputs_for_generation(self, input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        
        if "past_key_values_distill" in kwargs:
            model_inputs["past_key_values_distill"] = kwargs["past_key_values_distill"]
            
        return model_inputs


class OPTForDistill(OPTForCausalLM, BaseModelForDistill):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)


class GPT2ForDistill(GPT2LMHeadModel, BaseModelForDistill):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)


class FalconForDistill(FalconForCausalLM, BaseModelForDistill):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super().__init__(config)
        
        
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        
        model_kwargs = FalconForCausalLM._update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, standardize_cache_format)
        
        # update past_key_values
        past_key_values_distill = outputs.past_key_values_distill
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values_distill = self._convert_to_standard_cache(past_key_values_distill, batch_size=batch_size)
        model_kwargs["past_key_values_distill"] = past_key_values_distill
        return model_kwargs
        
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        model_inputs = FalconForCausalLM.prepare_inputs_for_generation(self, input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs)
        
        if "past_key_values_distill" in kwargs:
            model_inputs["past_key_values_distill"] = kwargs["past_key_values_distill"]
            
        return model_inputs


class PlausibleLogitsWarper(LogitsWarper):
    def __init__(self, alpha: float, filter_value: float = -float("Inf")):
        if not isinstance(alpha, float) or alpha <= 0 or alpha > 1:
            raise ValueError(f"`alpha` has to be a float > 0 and <= 1, but is {alpha}")

        self.alpha = alpha
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = scores.softmax(dim=-1)
        probs_max = probs.max(axis=-1, keepdim=True).values
        indices_to_remove = (probs < self.alpha*probs_max)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        return scores


class MinPLogitsWarper(LogitsWarper):
    def __init__(self, beta: float, min_tokens_to_keep: int = 1):
        beta = float(beta)
        if beta < 0 or beta > 1.0:
            raise ValueError(f"`beta` has to be a float > 0 and < 1, but is {beta}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.beta = beta
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative beta above the threshold (token with 0 are kept)
        sorted_indices_to_constrain = cumulative_probs <= (1 - self.beta)
        # Keep at least min_tokens_to_keep
        sorted_indices_to_constrain[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_constrain = sorted_indices_to_constrain.scatter(1, sorted_indices, sorted_indices_to_constrain)
        min_score = torch.min(scores[~indices_to_constrain])
        scores_processed = scores.masked_fill(indices_to_constrain, min_score)
        return scores_processed


class MinKLogitsWarper(LogitsWarper):
    def __init__(self, kappa: int, min_tokens_to_keep: int = 1):
        if not isinstance(kappa, int) or kappa <= 0:
            raise ValueError(f"`kappa` has to be a strictly positive integer, but is {kappa}")

        self.kappa = max(kappa, min_tokens_to_keep)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        kappa = min(self.kappa, scores.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_constrain = scores < torch.topk(scores, kappa)[0][..., -1, None]
        min_score = torch.min(scores[~indices_to_constrain])
        scores_processed = scores.masked_fill(indices_to_constrain, min_score)
        return scores_processed
