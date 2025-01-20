# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction, IntervalStrategy
import importlib
import sys

def is_peft_available():
    return importlib.util.find_spec("peft") is not None

# if is_peft_available():
#     from peft import PeftModel, get_peft_model, prepare_model_for_int8_training
    
import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase, TrainerCallback

@dataclass
class RewardDataCollatorWithPadding:
    r"""
    Reward DataCollator class that pads the inputs to the maximum length of the batch.
    Args:
        tokenizer (`PreTrainedTokenizerBase`):
            The tokenizer used for encoding the data.
        padding (`Union[bool, str, `PaddingStrategy`]`, `optional`, defaults to `True`):
            padding_strategy to pass to the tokenizer.
        max_length (`Optional[int]`, `optional`, defaults to `None`):
            The maximum length of the sequence to be processed.
        pad_to_multiple_of (`Optional[int]`, `optional`, defaults to `None`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`, `optional`, defaults to `"pt"`):
            The tensor type to use.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        for feature in features:
            # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            features_rejected.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch_chosen = self.tokenizer.pad(
            features_chosen,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_rejected = self.tokenizer.pad(
            features_rejected,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_chosen": batch_chosen["input_ids"],
            "attention_mask_chosen": batch_chosen["attention_mask"],
            "input_ids_rejected": batch_rejected["input_ids"],
            "attention_mask_rejected": batch_rejected["attention_mask"],
            "return_loss": True,
        }
        return batch

def compute_accuracy(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    # Here, predictions is rewards_chosen and rewards_rejected.
    # We want to see how much of the time rewards_chosen > rewards_rejected.
    predictions = np.argmax(predictions, axis=1)

    accuracy = np.array(predictions == labels, dtype=float).mean().item()
    return {"accuracy": accuracy}

def log_sigmoid(x):
    return torch.clamp(x, max=0) - torch.log(torch.exp(-torch.abs(x)) + 1) + 0.5 * torch.clamp(x, min=0, max=0)

def on_step_end(self, args, state, control, **kwargs):
        # Log
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if (
            args.eval_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step
        ):
            control.should_evaluate = True

        # Save
        if (
            args.save_strategy == IntervalStrategy.STEPS
            and state.save_steps > 0
            and state.global_step in args.save_steps_value
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Save the model at the end if we have a save strategy
            if args.save_strategy != IntervalStrategy.NO:
                control.should_save = True

        return control

class RewardTrainer_power_save(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`transformers.TrainingArguments`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        # if not is_peft_available() and peft_config is not None:
        #     raise ValueError(
        #         "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        #     )
        # elif is_peft_available() and peft_config is not None:
        #     if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
        #         model = prepare_model_for_int8_training(model)

        #     model = get_peft_model(model, peft_config)

        # if is_peft_available() and callbacks is None and isinstance(model, PeftModel):
        #     callbacks = [PeftSavingCallback()]

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `max_length` in the RewardTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        setattr(self.callback_handler, 'on_step_end', on_step_end.__get__(self.callback_handler))

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        out_dict = {}

        if self.args.rm_coef > 0:
            output = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True,
                output_hidden_states=True,
            )

            rewards = output.logits
            
            real_batch_size = inputs['input_ids'][::2].shape[0]
            fake_batch_size = inputs['input_ids'][1::2].shape[0]
            
            rewards_chosen = rewards[::2]
            rewards_rejected = rewards[1::2]
            
            rm_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
            out_dict.update({
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            })
        else:
            if hasattr(model, 'model'):
                output = model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            else:
                output = model.module.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            rm_loss = 0
        
        if self.args.lm_coef > 0:
            if self.args.rm_coef > 0:
                hidden_states = output.hidden_states[-1][::2].to(torch.bfloat16)
            else:
                hidden_states = output[0][::2].to(torch.bfloat16)
            labels = inputs['labels'][::2]
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden_states)
            else:
                logits = model.module.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                if hasattr(model, 'module'):
                    shift_logits = shift_logits.view(-1, model.module.config.vocab_size)
                else:
                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)

            out_dict.update({
                'lm_logits': logits,
            })
        else:
            lm_loss = 0

        loss = self.args.rm_coef * rm_loss + self.args.lm_coef * lm_loss 
        if return_outputs:
            return loss, out_dict
        return loss

class RewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`transformers.TrainingArguments`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        # if not is_peft_available() and peft_config is not None:
        #     raise ValueError(
        #         "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        #     )
        # elif is_peft_available() and peft_config is not None:
        #     if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
        #         model = prepare_model_for_int8_training(model)

        #     model = get_peft_model(model, peft_config)

        # if is_peft_available() and callbacks is None and isinstance(model, PeftModel):
        #     callbacks = [PeftSavingCallback()]

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `max_length` in the RewardTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        out_dict = {}

        if self.args.rm_coef > 0:
            output = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=True,
                output_hidden_states=True,
            )

            rewards = output.logits
            
            real_batch_size = inputs['input_ids'][::2].shape[0]
            fake_batch_size = inputs['input_ids'][1::2].shape[0]
            
            rewards_chosen = rewards[::2]
            rewards_rejected = rewards[1::2]
            
            rm_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
            out_dict.update({
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            })
        else:
            if hasattr(model, 'model'):
                output = model.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            else:
                output = model.module.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                )
            rm_loss = 0
        
        if self.args.lm_coef > 0:
            if self.args.rm_coef > 0:
                hidden_states = output.hidden_states[-1][::2].to(torch.bfloat16)
            else:
                hidden_states = output[0][::2].to(torch.bfloat16)
            labels = inputs['labels'][::2]
            if hasattr(model, 'lm_head'):
                logits = model.lm_head(hidden_states)
            else:
                logits = model.module.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                if hasattr(model, 'module'):
                    shift_logits = shift_logits.view(-1, model.module.config.vocab_size)
                else:
                    shift_logits = shift_logits.view(-1, model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                lm_loss = loss_fct(shift_logits, shift_labels)

            out_dict.update({
                'lm_logits': logits,
            })
        else:
            lm_loss = 0

        loss = self.args.rm_coef * rm_loss + self.args.lm_coef * lm_loss 
        if return_outputs:
            return loss, out_dict
        return loss