# +
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Dict, Any, Tuple
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

from dp_transformers.dp_utils import OpacusDPTrainer

from arguments import TrainingArguments, PrivacyArguments
from metrics import (
    compute_txt_metrics,
    compute_grouped_metrics,
)


# +
class CausalTrainer(Trainer):
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = super()._prepare_inputs(inputs)
        inputs = BatchEncoding({key: tensor for key, tensor in inputs.items() if key in ["input_ids", "attention_mask", "token_type_ids", "labels"]})
        return inputs
    
    def _prepare_gen_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        inputs = super()._prepare_inputs(inputs)
        gen_inputs = {
            "input_ids": inputs["gen_input_ids"],
            "attention_mask": inputs["gen_attention_mask"],
            "labels": inputs["gen_labels"],
        }
        if "gen_token_type_ids" in inputs: gen_inputs["token_type_ids"] = inputs["gen_token_type_ids"]
        return gen_inputs
    
    # rewrite the prediction step for decoder only LMs
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        model_inputs = self._prepare_inputs(inputs)        
        with torch.no_grad():
            if "labels" in inputs:
                with self.compute_loss_context_manager():
                    outputs = model(**model_inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, model_inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None
        
        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # gen_kwargs > model.generation_config > default GenerationConfig()
        gen_kwargs = {
            "max_new_tokens": self.args.generation_max_new_tokens,
            "num_beams": self.args.generation_num_beams,
        }
        if gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
            gen_kwargs["max_length"] = (
                gen_kwargs["max_length"] if gen_kwargs.get("max_length") is not None else self.model.config.max_length
            )
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.model.config.num_beams
        )
        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )
        
        gen_inputs = self._prepare_gen_inputs(inputs)
        generated_tokens = self.model.generate(**gen_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

#         # Retrieves GenerationConfig from model.generation_config
#         gen_config = self.model.generation_config
#         # in case the batch is shorter than max length, the output should be padded
#         if generated_tokens.shape[-1] < gen_config.max_length:
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
#         elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
#             generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        labels = gen_inputs["labels"] if "labels" in gen_inputs else None

        return loss, generated_tokens, labels
    

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = CausalTrainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            print(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    print(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        sgd_kwargs = {
            "momentum": args.momentum,
            "dampening": args.momentum,
        }
        rmsprop_kwargs = {
            "alpha": args.rms_alpha,
            "eps": args.rms_eps,
        }
        if args.rmsprop:
            optimizer_cls = torch.optim.RMSprop
            optimizer_kwargs.update(rmsprop_kwargs)
        elif args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        elif args.optim == OptimizerNames.ADAMW_BNB:
            try:
                from bitsandbytes.optim import Adam8bit

                optimizer_cls = Adam8bit
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate bnb Adam8bit but bnb is not installed!")
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
            optimizer_kwargs.update(sgd_kwargs)
        elif args.optim == OptimizerNames.ADAGRAD:
            optimizer_cls = torch.optim.Adagrad
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs
    
    # rewrite the evaluation loop to compute loss for each dataset
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        args = self.args
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f" Max batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.
        
        all_datasets = None

        observed_num_examples = 0
        # Main evaluation loop
        
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            if "dataset" in inputs: 
                all_datasets = all_datasets + inputs["dataset"] if all_datasets is not None else inputs["dataset"]
                
            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(observed_batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=self.tokenizer.pad_token_id)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, inputs)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=self.tokenizer.pad_token_id)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=self.tokenizer.pad_token_id)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=self.tokenizer.pad_token_id)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=self.tokenizer.pad_token_id)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=self.tokenizer.pad_token_id)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        if all_datasets is not None:
            all_datasets = all_datasets[:num_samples]
            
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(trainer=self, dataset=eval_dataset, preds=all_preds, labels=all_labels, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        if self.lr_scheduler is not None:
            metrics["lr"] = self._get_learning_rate()
            
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            for dataset in np.unique(all_datasets):
                losses = all_losses[np.where(np.array(all_datasets) == dataset)]
                metrics[f"{metric_key_prefix}_loss_{dataset}"] = losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            raise NotImplementedError
        
        eval_batch_sampler = TaskGroupedBatchSampler(batch_size=self.args.per_device_eval_batch_size, dataset=eval_dataset)    
        dataloader = DataLoader(
            eval_dataset,
            batch_sampler=eval_batch_sampler, 
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        return dataloader
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        super().save_model(output_dir, _internal_call)
        if self.args.peft is not None:
            if output_dir is None: output_dir = self.args.output_dir            
            self.model.save_pretrained(output_dir)
            
    def compute_losses(self, model, inputs):
        outputs = model(**inputs)
        
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
    
    def compute_unnorm_losses(self, model, inputs):
        outputs = model(**inputs)
        
        logits_target = outputs.logits_target
        logits_distill = outputs.logits_distill
        logits = logits_target - logits_distill
        labels = inputs["labels"]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        losses_seq = -torch.gather(shift_logits, dim=-1, index=shift_labels[:, :, None]).squeeze(-1)
        mask_labels = shift_labels != self.tokenizer.pad_token_id
        losses = torch.sum(losses_seq * mask_labels, -1) / mask_labels.sum(-1)

        return losses
    
    def compute_uniform_losses(self, model, inputs):
        outputs = model(**inputs)

        logits = outputs.logits
        labels = inputs["labels"]
        uniforms = torch.ones_like(logits) / logits.size(-1)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_uniforms = uniforms[..., 1:, :].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")

        losses_flat = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_uniforms.view(-1, model.config.vocab_size))
        losses_seq = losses_flat.view(shift_labels.shape)
        mask_labels = shift_labels != self.tokenizer.pad_token_id
        losses = torch.sum(losses_seq * mask_labels, -1) / mask_labels.sum(-1)

        return losses


# -

class DPTrainer(OpacusDPTrainer, CausalTrainer):
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        privacy_args: PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Dict
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            privacy_args=privacy_args,
            author_mapping=author_mapping,
            **kwargs,
        )
        
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        OpacusDPTrainer.save_model(self, output_dir, _internal_call)
        if self.args.peft is not None:
            if output_dir is None: output_dir = self.args.output_dir
            self.model._module.save_pretrained(output_dir)


class TaskGroupedBatchSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        dataset: Optional[Dataset] = None,
        shuffle = False,
        generator = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size

        self.generator = generator
        
        dataset = dataset.flatten_indices()
        dataset = dataset.add_column("indices", np.arange(len(dataset)))
        mega_batches = [dataset.filter(lambda example: example["dataset"] == dataset_name)["indices"] for dataset_name in dataset.unique("dataset")]
        self.batches = []
        for mega_batch in mega_batches:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in mega_batch:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    self.batches.append(batch)
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                self.batches.append(batch[:idx_in_batch])
        
        if shuffle: random.shuffle(self.batches)
        
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


class DenserEvalCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        return control
