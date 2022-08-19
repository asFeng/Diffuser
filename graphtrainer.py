
import dgl
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, Callable, List, Dict
from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from models.diffuser_utils import DiffuserConfig

class graphTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        config: DiffuserConfig = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__(model,
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
        self.config = config
        self._create_adj_mat()

    def _create_adj_mat(self):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        max_len = 4096 # not the input sequence max len
        n_blocks = max_len//(attention_window//2)-1
        adj = np.zeros([max_len, max_len])
        
        # add local window att (overlap)
        for i in range(n_blocks):
            start = i*attention_window//2
            end = start+attention_window
            if end > max_len:
                end = max_len
            adj[start:end, start:end] = 1

        # add random att    
        np.random.seed(0)
        num_random = max_len*self.config.num_rand
        
        idx = np.random.choice(range(max_len*max_len), num_random ,replace=False)
        idx_x = idx %  max_len
        idx_y = idx // max_len
        adj[idx_x,idx_y] = 1

        # add global att    
        num_global = self.config.num_glob
        idx = np.random.choice(range(attention_window,max_len), num_global ,replace=False)
        adj[idx,:] = 1
        adj[:,idx] = 1

        possible_seq_len = np.arange(attention_window, max_len+attention_window, attention_window)
        self.src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}

    def _pad_to_window_size(self,inputs):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = inputs["input_ids"].shape if inputs["input_ids"] is not None else inputs["attention_mask"].shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logging.debug(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if inputs["input_ids"] is not None:
                inputs["input_ids"] = nn.functional.pad(inputs["input_ids"], (0, padding_len), value=self.config.pad_token_id)
            inputs["attention_mask"] = nn.functional.pad(
                inputs["attention_mask"], (0, padding_len), value=False
            )  # no attention on the padding tokens
        return inputs

    def _from_adj_to_batched_graphs(self, input_ids):
        B = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        g_list = []
        for i in range(B):
            src,dst =self.src_dst[seq_len]
            g = dgl.graph((src, dst))
            g_list.append(g)
        batched_g = dgl.batch(g_list)
        return batched_g  

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = self._pad_to_window_size(inputs) 
        device =inputs["input_ids"].device
        batched_g = self._from_adj_to_batched_graphs(inputs["input_ids"]).to(device)
        inputs["g"] = batched_g
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss