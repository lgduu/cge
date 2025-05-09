# +
import pdb
import os
import sys
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Dict, Any, Tuple, Callable
import random
import math
import time
import copy
import json
from itertools import islice

from tqdm import tqdm
from packaging import version
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.sampler import Sampler
import datasets
from datasets import Dataset

from transformers import is_datasets_available
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import Trainer, logger, skip_first_batches, TRAINER_STATE_NAME, accelerate_version
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.debug_utils import DebugOption
from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_utils import (
    TrainOutput,
    EvalLoopOutput,
    EvalPrediction,
    has_length,
    is_torch_tpu_available,
    denumpify_detensorize,
    ShardedDDPOption,
    seed_worker,
    speed_metrics,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_pt_utils import (
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    find_batch_size,
    get_parameter_names,
    get_model_param_count,
)
from transformers.deepspeed import is_deepspeed_zero3_enabled
# from transformers.trainer import _is_torch_generator_available
from transformers.optimization import Adafactor, get_scheduler

from causal_trainer import CausalTrainer


# +
class ContrastTrainer(CausalTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        model_target: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if model_target is not None: 
            self.model_target = model_target
            

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            
        if self.args.unlearn:
            loss = -loss

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
            
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
#         self._train_batch_size = batch_size
#         logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
#         # Data loader and number of training steps
#         train_dataloader = self.get_train_dataloader()

#         # Setting up training control variables:
#         # number of training epochs: num_train_epochs
#         # number of training steps per epoch: num_update_steps_per_epoch
#         # total number of training steps to execute: max_steps
#         total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

#         len_dataloader = None
#         num_train_tokens = None
#         if has_length(train_dataloader):
#             len_dataloader = len(train_dataloader)
#             num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
#             num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
#             num_examples = self.num_examples(train_dataloader)
#             if args.max_steps > 0:
#                 max_steps = args.max_steps
#                 num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
#                     args.max_steps % num_update_steps_per_epoch > 0
#                 )
#                 # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
#                 # the best we can do.
#                 num_train_samples = args.max_steps * total_train_batch_size
#                 if args.include_tokens_per_second:
#                     num_train_tokens = (
#                         self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
#                     )
#             else:
#                 max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
#                 num_train_epochs = math.ceil(args.num_train_epochs)
#                 num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
#                 if args.include_tokens_per_second:
#                     num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
#         elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
#             max_steps = args.max_steps
#             # Setting a very large number of epochs so we go as many times as necessary over the iterator.
#             num_train_epochs = sys.maxsize
#             num_update_steps_per_epoch = max_steps
#             num_examples = total_train_batch_size * args.max_steps
#             num_train_samples = args.max_steps * total_train_batch_size
#             if args.include_tokens_per_second:
#                 num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
#         else:
#             raise ValueError(
#                 "args.max_steps must be set to a positive value if dataloader does not have a length, was"
#                 f" {args.max_steps}"
#             )

        ## addition start ##
        if self.train_dataset is not None: train_dataloader = iter(self.get_train_dataloader())
        total_train_batch_size = args.distill_num * args.gradient_accumulation_steps * args.world_size
        
        num_train_tokens = None
        if args.distill_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.distill_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = args.distill_length * args.distill_num * args.max_steps * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.distill_steps}"
            )
        ## addition end ##

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
            or self.is_fsdp_enabled
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
#         self.callback_handler.train_dataloader = train_dataloader

        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                is_random_sampler = isinstance(sampler, RandomSampler)
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        epoch = 0
#         for epoch in range(epochs_trained, num_train_epochs):
#             epoch_iterator = train_dataloader

        # Reset the past mems state at the beginning of each epoch if necessary.
        if args.past_index >= 0:
            self._past = None

        ## addition start ##
#         steps_in_epoch = (
#             len(epoch_iterator)
#             if len_dataloader is not None
#             else args.max_steps * args.gradient_accumulation_steps
#         )
        steps_in_epoch = max_steps * args.gradient_accumulation_steps
        ## addition end ##

        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

        if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            self._load_rng_state(resume_from_checkpoint)

        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True

        step = -1
        
        ## addition start ##
        if self.model.config.model_type == "gpt2" or self.model.config.model_type == "falcon":
            bos_inputs = self.tokenizer([self.tokenizer.bos_token], return_tensors="pt")
        else:
            bos_inputs = self.tokenizer([""], return_tensors="pt")
        bos_inputs = self._prepare_inputs(bos_inputs)
        
        self.step_distills = {}
        all_distills = None
        
#         for step, inputs in enumerate(epoch_iterator):
        for step in range(steps_in_epoch):
        ## addition end ##
        
            total_batched_samples += 1
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                if steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)
                continuei
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None

            if step % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

            ## addition start ##
            if self.args.baseline:
                assert self.args.per_device_train_batch_size == self.args.distill_num
                inputs = next(train_dataloader)
                distill_ids = inputs["input_ids"]
            else:
                if self.args.contrast:
                    distill_ids = self.generate_distills(
                        inputs=bos_inputs,
                        alpha=self.args.alpha,
                        beta=self.args.beta,
                        kappa=self.args.kappa,
                        gamma=self.args.gamma,
                        max_length=self.args.distill_length,
                    )
                elif self.args.extract:
                    distill_ids = self.extract_distills(
                        train_dataloader=train_dataloader,
                        gamma=self.args.gamma,
                    )

                inputs = {
                    "input_ids": distill_ids,
                    "attention_mask": (distill_ids != self.tokenizer.pad_token_id).to(torch.int64),
                    "labels": distill_ids,
                }
                
            all_distills = distill_ids if all_distills is None else nested_concat(all_distills, distill_ids, padding_index=self.tokenizer.pad_token_id)
            
            ## addition end ##
                
            with self.accelerator.accumulate(model):
                tr_loss_step = self.training_step(model, inputs)

            if (
                args.logging_nan_inf_filter
                and not is_torch_tpu_available()
                and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
            ):
                # if loss is nan or inf simply add the average of previous logged losses
                tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                tr_loss += tr_loss_step

            self.current_flos += float(self.floating_point_ops(inputs))

            is_last_step_and_steps_less_than_grad_acc = (
                steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
            )

            if (
                total_batched_samples % args.gradient_accumulation_steps == 0
                or
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                is_last_step_and_steps_less_than_grad_acc
            ):
                # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                # in accelerate. So, explicitly enable sync gradients to True in that case.
                if is_last_step_and_steps_less_than_grad_acc or (
                    version.parse(accelerate_version) <= version.parse("0.20.3")
                ):
                    self.accelerator.gradient_state._set_sync_gradients(True)

                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping

                    if self.do_grad_scaling:
                        # Reduce gradients first for XLA
                        if is_torch_tpu_available():
                            gradients = xm._fetch_gradients(self.optimizer)
                            xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(self.optimizer)

                    if is_sagemaker_mp_enabled() and args.fp16:
                        self.optimizer.clip_master_grads(args.max_grad_norm)
                    elif hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    elif self.use_apex:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer),
                            args.max_grad_norm,
                        )
                    else:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                # Optimizer step
                optimizer_was_run = True
                if is_torch_tpu_available():
                    if self.do_grad_scaling:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                        self.optimizer.step()
                elif self.do_grad_scaling:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                if optimizer_was_run:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                if not self.args.accumulate: model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                ## addition start ##
                if all_distills is not None: 
                    self.step_distills[self.state.global_step-1] = all_distills
                    print(f"step{self.state.global_step-1}\n", self.tokenizer.batch_decode(all_distills), "\n")
                    
                all_distills = None
                ## addition end ##
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            else:
                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break
                
        if step < 0:
            logger.warning(
                "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        ## addition start ##
        if all_distills is not None: 
            self.step_distills[self.state.global_step-1] = all_distills
            print(f"step{self.state.global_step-1}\n", self.tokenizer.batch_decode(all_distills), "\n")
            
        all_distills = None
        ## addition end ##
        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_tpu_available():
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
#         if self.control.should_training_stop:
#             break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.save_distills()
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def compute_losses_distill(self, model, inputs):
        outputs = model.forward_distill(**inputs)
        
        logits = outputs.logits
        labels = inputs["labels"]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")

        losses_flat = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
        losses_seq = losses_flat.view(shift_labels.shape)
        mask_labels = shift_labels != self.tokenizer.pad_token_id
        losses = torch.sum(losses_seq * mask_labels, -1) / mask_labels.sum(-1)

        return losses
    
    def compute_unnorm_losses_distill(self, model, inputs):
        outputs = model.forward_distill(**inputs)
        
        logits_target = outputs.logits_target
        logits_distill = outputs.logits_distill
        logits = logits_target - model.gamma*logits_distill
        labels = inputs["labels"]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses_seq = -torch.gather(shift_logits, dim=-1, index=shift_labels[:, :, None]).squeeze(-1)
        mask_labels = shift_labels != self.tokenizer.pad_token_id
        losses = torch.sum(losses_seq * mask_labels, -1) / mask_labels.sum(-1)

        return losses
    
    def extract_distills(
        self,
        train_dataloader,
        gamma,
    ):
        self.model_target.gamma = gamma
        self.model.eval()
        
        all_candidate_ids = None
        all_candidate_losses = None
        
        for _ in range(self.args.candidate_num):
            candidate_inputs = next(train_dataloader)
            candidate_ids = candidate_inputs["input_ids"]
            with torch.no_grad():
                if self.args.unnorm:
                    candidate_losses = self.compute_unnorm_losses_distill(self.model_target, candidate_inputs)
                else:
                    candidate_losses = self.compute_losses_distill(self.model_target, candidate_inputs)

            all_candidate_ids = candidate_ids if all_candidate_ids is None else nested_concat(all_candidate_ids, candidate_ids, padding_index=self.tokenizer.pad_token_id)
            all_candidate_losses = candidate_losses if all_candidate_losses is None else torch.cat((all_candidate_losses, candidate_losses), dim=0)
            
        distill_indices = all_candidate_losses.sort().indices[:self.args.distill_num]
        distill_ids = all_candidate_ids[distill_indices]
        
        self.model.train()
        return distill_ids

    def generate_distills(
        self, 
        inputs, 
        alpha,
        beta,
        kappa,
        gamma,
        max_length=None,
        max_new_tokens=None,
        num_return_sequences=None,
        do_sample=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        top_k=None,
        top_p=None,
        typical_p=None,
        output_scores=False,
    ):
        assert not (beta > 0 and kappa > 0)
        
        model_distill = self.model if self.args.unlearn else self.model_target
        
        model_distill.alpha = alpha
        model_distill.beta = beta
        model_distill.kappa = kappa
        model_distill.gamma = gamma
        model_distill.get_logits_warper()
                
        assert max_length or max_new_tokens
        
        self.model.eval()
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.args.distill_num_beam_groups
        if num_beam_groups > 1:
            distill_outputs = model_distill.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=max_new_tokens, 
                num_return_sequences=num_return_sequences if num_return_sequences is not None else self.args.distill_num, 
                do_sample=do_sample if do_sample is not None else self.args.distill_do_sample, 
                num_beams=num_beams if num_beams is not None else self.args.distill_num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty if diversity_penalty is not None else self.args.distill_diversity_penalty,
                top_k=top_k if top_k is not None else self.args.distill_top_k,
                top_p=top_p if top_p is not None else self.args.distill_top_p,
                typical_p=typical_p if typical_p is not None else self.args.distill_typical_p, 
                output_scores=True,
                return_dict_in_generate=True,
            )
            distill_ids = distill_outputs.group_sequences
        else:
            distill_ids = model_distill.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=max_new_tokens, 
                num_return_sequences=num_return_sequences if num_return_sequences is not None else self.args.distill_num, 
                do_sample=do_sample if do_sample is not None else self.args.distill_do_sample, 
                num_beams=num_beams if num_beams is not None else self.args.distill_num_beams,
                num_beam_groups=num_beam_groups,
                diversity_penalty=diversity_penalty if diversity_penalty is not None else self.args.distill_diversity_penalty,
                top_k=top_k if top_k is not None else self.args.distill_top_k,
                top_p=top_p if top_p is not None else self.args.distill_top_p,
                typical_p=typical_p if typical_p is not None else self.args.distill_typical_p, 
                output_scores=output_scores,
                return_dict_in_generate=True if output_scores else False,
            )
        self.model.train()
        return distill_ids
    
    def save_distills(self):
        distill_path = os.path.join(self.args.output_dir, self.args.distill_path)
        step_distills = {epoch: all_distills.tolist() for epoch, all_distills in self.step_distills.items()}
        with open(distill_path, "w", encoding="utf-8") as f:
            json_string = json.dumps(step_distills, sort_keys=True)
            f.write(json_string)
            
    def sample_distills(self):
        if self.model.config.model_type == "gpt2" or self.model.config.model_type == "falcon":
            bos_inputs = self.tokenizer([self.tokenizer.bos_token], return_tensors="pt")
        else:
            bos_inputs = self.tokenizer([""], return_tensors="pt")
        bos_inputs = self._prepare_inputs(bos_inputs)
        self.step_distills = {}
        for step in tqdm(range(self.args.distill_steps)):
            distill_ids = self.generate_distills(
                inputs=bos_inputs,
                alpha=self.args.alpha,
                beta=self.args.beta,
                kappa=self.args.kappa,
                gamma=self.args.gamma,
                max_length=self.args.distill_length,
            )
            self.step_distills[step] = distill_ids
            print(f"step{step}\n", self.tokenizer.batch_decode(distill_ids), "\n")
        self.save_distills()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_save: self.save_distills()
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
# -




