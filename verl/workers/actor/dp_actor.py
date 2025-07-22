# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import (
    gather_outputs_and_unpad,
    ulysses_pad_and_slice_inputs,
    slice_input_tensor,
    get_ulysses_sequence_parallel_rank,
    get_ulysses_sequence_parallel_group,
    _ULYSSES_SEQUENCE_PARALLEL_GROUP,
)
from .base import BasePPOActor
from .config import ActorConfig

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass

import torch.nn.functional as F
import torch.distributed as dist
from verl.utils.vila_remote_code.constants import IGNORE_INDEX

__all__ = ["DataParallelPPOActor"]
NUM_TOKENS_PER_IMAGE = 257  # (256+1)


def extract_local_from_list(value_list, sp_rank, sp_size):
    quotient, remainder = divmod(len(value_list), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    return value_list[start_idx:end_idx]

def left_pad(x, target_len, dim=-1, value=0):
    pad_len = target_len - x.size(dim)
    if pad_len <= 0:
        return x
    pad = [0] * (2 * x.ndim)
    pad[-2 * dim - 1] = pad_len
    return F.pad(x, pad, value=value)

def prepare_inputs_for_sp(inputs: torch.Tensor, attention_mask: torch.Tensor, response_length: int, position_ids: torch.Tensor=None,
                          responses: torch.Tensor=None, sp_size: int = 1, sp_rank: int = 0, padding_id: int = -1):
    input_text_length = inputs.size(1)
    assert input_text_length % sp_size == 0, "input_text_length must be divisible by sp_size"
    seqlens_in_batch = inputs.size(1)

    sp_middle_rank_len = input_text_length // sp_size
    padded_inputs = inputs
    local_responses = None

    start_index = sp_middle_rank_len * sp_rank
    end_index = sp_middle_rank_len * (sp_rank + 1)
    local_inputs = padded_inputs[:, start_index : end_index]
    local_attention_mask = attention_mask[:, start_index : end_index]
    if responses is not None:
        local_responses = responses[:, start_index : end_index]
    if position_ids.dim() == 3:
        local_position_ids = position_ids[:, :, start_index : end_index]
    else:
        local_position_ids = position_ids[:, start_index : end_index]

    return local_inputs, local_attention_mask, local_position_ids.contiguous(), local_responses, seqlens_in_batch

def prepare_inputs_for_sp_mm(inputs: torch.Tensor, attention_mask: torch.Tensor, multi_modal_labels: torch.Tensor, response_length: int, position_ids: torch.Tensor=None,
                          responses: torch.Tensor=None, sp_size: int = 1, sp_rank: int = 0, visual_id: int = IGNORE_INDEX, padding_id: int = -1, vila_model: bool = False):
    image_labels = multi_modal_labels == visual_id
    video_token_num = image_labels[0].sum()
    input_text_length = image_labels.size(1) - video_token_num

    #assert inputs.dim() == 3, "inputs should be 2D (batch_size, seq_len, num_dim)"
    batch_size = inputs.size(0)
    padded_inputs = inputs
    seqlens_in_batch = inputs.size(1)

    # Step 1: Generate global position_ids
    if position_ids is None:
        padding_labels = multi_modal_labels != padding_id
        padding_labels = F.pad(padding_labels, (0, response_length), value=True)
        first_true_idx = torch.argmax(padding_labels.int(), dim=1)
        row_idx = torch.arange(padding_labels.size(1), device=multi_modal_labels.device).unsqueeze(0).expand(padding_labels.size(0), -1)
        relative_idx = row_idx - first_true_idx.unsqueeze(1)
        position_ids = torch.where(padding_labels, relative_idx, torch.full_like(relative_idx, -1))

    # Step 2: Find video token position and validate
    video_token_indice = image_labels[0].nonzero(as_tuple=True)
    assert len(video_token_indice) == 1, f"Sample requires exactly one video token {video_token_indice}"
    video_token_indice = video_token_indice[0]
    all_match = image_labels[:, video_token_indice].all()
    assert all_match, "Video token in all samples should be the same location, i.e., after system prompt"

    # Step 3: Generate all components based on sp_rank
    video_token_id_first = video_token_indice[0]
    video_token_id_last = video_token_indice[-1]
    sp_middle_rank_len = video_token_num // sp_size
    output_length = max(input_text_length, response_length) + sp_middle_rank_len

    local_responses = None
    if sp_rank == 0:
        # First rank: tokens up to and including video token
        end_index = video_token_id_first + sp_middle_rank_len
        local_inputs = padded_inputs[:, : end_index]
        local_attention_mask = attention_mask[:, : end_index]
        if responses is not None:
            local_responses = responses[:, : end_index]
        if position_ids.dim() == 3:
            local_position_ids = position_ids[:, :, : end_index]
        else:
            local_position_ids = position_ids[:, : end_index]
    elif sp_rank == sp_size - 1:
        # Last rank: tokens from video token to end
        if vila_model:
            start_index = -(sp_middle_rank_len + response_length)
        else:
            start_index = video_token_id_last + 1 - sp_middle_rank_len
        local_inputs = padded_inputs[:, start_index:]
        local_attention_mask = attention_mask[:, start_index: ]
        if responses is not None:
            local_responses = responses[:, start_index:]
        if position_ids.dim() == 3:
            local_position_ids = position_ids[:, :, start_index :]
        else:
            local_position_ids = position_ids[:, start_index :]
    else:
        # Middle ranks: only video token
        start_index = video_token_id_first + sp_middle_rank_len * sp_rank
        end_index = video_token_id_first + sp_middle_rank_len * (sp_rank + 1)
        local_inputs = padded_inputs[:, start_index : end_index]
        local_attention_mask = attention_mask[:, start_index : end_index]
        if responses is not None:
            local_responses = responses[:, start_index : end_index]
        if position_ids.dim() == 3:
            local_position_ids = position_ids[:, :, start_index : end_index]
        else:
            local_position_ids = position_ids[:, start_index : end_index]

    # Step 4: Pad to output_length on dim=1
    pad_len = output_length - local_inputs.size(1)
    if pad_len > 0:
        if local_inputs.dim() == 2:
            local_inputs = F.pad(local_inputs, (pad_len, 0), value=padding_id)
        else:
            local_inputs = F.pad(local_inputs, (0, 0, pad_len, 0), value=0.0)
        local_responses = F.pad(local_responses, (pad_len, 0), value=-100)
        local_attention_mask = F.pad(local_attention_mask, (pad_len, 0), value=False)
        local_position_ids = F.pad(local_position_ids, (pad_len, 0), value=-1)

    return local_inputs, local_attention_mask, local_position_ids.contiguous(), local_responses, seqlens_in_batch


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        model_vision_encoder=None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.model_vision_encoder = model_vision_encoder
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

        self.vila_model = config.vila_model
        if config.vila_model:
            self._forward_micro_batch = self._forward_micro_batch_vila
        else:
            self._forward_micro_batch = self._forward_micro_batch_ori

    def _forward_micro_batch_ori(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            for input_dict in micro_batch["multi_modal_inputs"]:
                for key, value in input_dict.items():
                    multi_modal_inputs[key].append(value)

            for key, value in multi_modal_inputs.items():
                if len(value) != 0:
                    multi_modal_inputs[key] = torch.cat(value, dim=0)
                else:
                    multi_modal_inputs[key] = None

        if self.config.padding_free and self.config.ulysses_size <= 1:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                position_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                    position_ids_rmpad, None, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )
            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            if self.config.ulysses_size > 1:
                sp_rank = get_ulysses_sequence_parallel_rank()
                padding_token_id = self.actor_module.config.bos_token_id
                responses_padded = F.pad(responses, (input_ids.size(-1) - response_length, 0), value=-100)
                if len(multi_modal_inputs):
                    if "pixel_values_images" in multi_modal_inputs:
                        multi_modal_key ="image"
                        visual_token_id = self.actor_module.config.image_token_id
                    elif "pixel_values_videos" in multi_modal_inputs:
                        multi_modal_key = "video"
                        visual_token_id = self.actor_module.config.video_token_id
                    else:
                        raise ValueError("Not supported other visual modality.")
                    local_input_ids, local_attention_mask, local_position_ids, local_responses, seqlens_in_batch = (
                        prepare_inputs_for_sp_mm(input_ids, attention_mask, input_ids, response_length, position_ids, responses_padded,
                                              sp_size=self.config.ulysses_size, sp_rank=sp_rank, visual_id=visual_token_id, padding_id=padding_token_id))
                    pixel_values = multi_modal_inputs.pop("pixel_values_%ss"%multi_modal_key)
                    grid_thw = multi_modal_inputs.pop("%s_grid_thw"%multi_modal_key)
                    local_pixel_values = pixel_values.chunk(self.config.ulysses_size, dim=0)[sp_rank]
                    local_grid_thw = grid_thw.clone()
                    local_grid_thw[:, 0] //= self.config.ulysses_size
                    multi_modal_inputs["pixel_values_%ss"%multi_modal_key] = local_pixel_values
                    multi_modal_inputs["%s_grid_thw"%multi_modal_key] = local_grid_thw
                else:
                    local_input_ids, local_attention_mask, local_position_ids, local_responses, seqlens_in_batch = (
                        prepare_inputs_for_sp(input_ids, attention_mask, response_length, position_ids, responses_padded,
                                              sp_size=self.config.ulysses_size, sp_rank=sp_rank, padding_id=padding_token_id))
            else:
                local_input_ids = input_ids
                local_attention_mask = attention_mask
                local_position_ids = position_ids
                local_responses = responses

            output = self.actor_module(
                input_ids=local_input_ids,
                attention_mask=local_attention_mask,
                position_ids=local_position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                if len(multi_modal_inputs):
                    log_probs = self.log_probs_from_logits(logits[:, -response_length - 1: -1], local_responses[:, -response_length:])  # (bsz, response_length) #
                else:
                    logits_padded = F.pad(logits, (0, 0, 0, 1), value=0.0)
                    local_responses_padded = F.pad(local_responses, (1, 0), value=-100)
                    log_probs = self.log_probs_from_logits(logits=logits_padded, labels=local_responses_padded)
                    log_probs = log_probs[:, : -1].contiguous()

                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=1, unpad_dim=1)
                log_probs = log_probs[:, -response_length:]
            else:
                logits = logits[:, -response_length - 1: -1, :]  # (bsz, response_length, vocab_size)
                log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _forward_micro_batch_vila(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        attention_mask = micro_batch["attention_mask"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)

        with torch.no_grad():
            dummy_input = torch.tensor([[1] * self.config.ulysses_size], dtype=torch.int64, device=self.actor_module.device)
            dummy_attention_mask = torch.tensor([[1] * self.config.ulysses_size], dtype=torch.bool, device=self.actor_module.device)
            _ = self.actor_module(input_ids=dummy_input, attention_mask=dummy_attention_mask)
        response_embeds = self.actor_module.model.embed_tokens(responses)
        multi_modal_embeds = torch.from_numpy(micro_batch["multi_modal_embeds"]).to(device=response_embeds.device, dtype=response_embeds.dtype)
        multi_modal_labels = torch.from_numpy(micro_batch["multi_modal_labels"]).to(device=response_embeds.device, dtype=response_embeds.dtype)
        attention_mask_embeds = multi_modal_labels != -1
        multi_modal_embeds = torch.cat([multi_modal_embeds, response_embeds], dim=1)
        attention_mask = torch.cat([attention_mask_embeds, attention_mask[:, -response_length:]], dim=1)

        # multi_modal_labels==IGNORE_INDEX: image embeds
        # multi_modal_labels==1: text embeds
        # multi_modal_labels==-1: padding
        # pad and slice the inputs if sp > 1
        if self.config.ulysses_size > 1:
            sp_rank = get_ulysses_sequence_parallel_rank()
            responses_padded = F.pad(responses, (attention_mask.size(-1) - response_length, 0), value=-100)
            local_multi_modal_embeds, local_attention_mask, local_position_ids, local_responses, seqlens_in_batch = (
                prepare_inputs_for_sp_mm(multi_modal_embeds, attention_mask, multi_modal_labels, response_length,
                                         responses=responses_padded, sp_size=self.config.ulysses_size, sp_rank=sp_rank, vila_model=True))
        else:
            local_multi_modal_embeds = multi_modal_embeds
            local_attention_mask = attention_mask
            local_position_ids = None
            local_responses = responses

        output = self.actor_module(
            inputs_embeds=local_multi_modal_embeds,
            attention_mask=local_attention_mask,
            position_ids=local_position_ids,
            use_cache=False,
            logits_to_keep=response_length + 1, # 2049
        )

        logits: torch.Tensor = output.logits
        logits.div_(temperature)

        if self.config.ulysses_size > 1:
            log_probs = self.log_probs_from_logits(logits[:, -response_length - 1: -1],
                                                   local_responses[:, -response_length:])  # (bsz, response_length) #

            # gather and unpad for the ulysses sp
            log_probs = gather_outputs_and_unpad(log_probs, gather_dim=1, unpad_dim=1)

            log_probs = log_probs[:, -response_length:]
        else:
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _forward_micro_batch_diffusion(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        # print("***micro_batch***", micro_batch)
        prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob_flow_grpo(self.actor_module,
                                                                                        self.scheduler,
                                                                                        micro_batch,
                                                                                        0,
                                                                                        micro_batch["prompt_embeds"],
                                                                                        micro_batch["pooled_prompt_embeds"] if "pooled_prompt_embeds" in micro_batch else None,
                                                                                        micro_batch["negative_prompt_embeds"] if "negative_prompt_embeds" in micro_batch else None,
                                                                                        micro_batch["negative_pooled_prompt_embeds"] if "negative_pooled_prompt_embeds" in micro_batch else None,
                                                                                        self.config)

        return log_prob, prev_sample_mean

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_embeds", "multi_modal_labels"] if self.vila_model else ["multi_modal_inputs"]

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs

    @torch.no_grad()
    def compute_log_prob_diffusion(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = 0.0
        select_keys = ["latents", "next_latents", "timesteps", "prompt_embeds", "pooled_prompt_embeds",
                       "negative_prompt_embeds", "negative_pooled_prompt_embeds"]
        non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        prev_sample_mean_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs, prev_sample_mean = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)
            prev_sample_mean_lst.append(prev_sample_mean)

        log_probs = torch.concat(log_probs_lst, dim=0)
        prev_sample_mean = torch.concat(prev_sample_mean_lst, dim=0)
        return log_probs, prev_sample_mean

    def update_policy(self, data: DataProto, rollout_data: DataProto = None) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_embeds", "multi_modal_labels"] if self.vila_model else ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        mini_batches_rollout = rollout_data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)
        
        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for (mini_batch, mini_batch_rollout) in zip(mini_batches, mini_batches_rollout):
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                micro_batches_rollout = mini_batch_rollout.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for (micro_batch, micro_batch_rollout) in zip(micro_batches, micro_batches_rollout):
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    
                    model_inputs_rollout = {**micro_batch_rollout.batch, **micro_batch_rollout.non_tensor_batch}
                    responses_rollout = model_inputs_rollout["responses"]
                    response_length_rollout = responses_rollout.size(1)
                    attention_mask_rollout = model_inputs_rollout["attention_mask"]
                    response_mask_rollout = attention_mask_rollout[:, -response_length_rollout:]
                    old_log_probs_rollout = model_inputs_rollout["old_log_probs"]
                    advantages_rollout = model_inputs_rollout["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    log_probs_rollout = self._forward_micro_batch(model_inputs_rollout, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    pg_loss_rollout, pg_metrics_rollout = compute_policy_loss(
                        old_log_probs=old_log_probs_rollout,
                        log_probs=log_probs_rollout,
                        advantages=advantages_rollout,
                        response_mask=response_mask_rollout,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        ref_log_probs_rollout = model_inputs_rollout["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kld_rollout = compute_kl(
                            log_probs=log_probs_rollout,
                            ref_log_probs=ref_log_probs_rollout,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        kl_loss_rollout = average_loss(kld_rollout, response_mask_rollout, mode=self.config.loss_avg_mode)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        pg_loss_rollout = pg_loss_rollout + kl_loss_rollout * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                        metrics["actor/kl_loss_rollout"] = kl_loss_rollout.detach().item()
                        metrics["actor/kl_coef_rollout"] = self.config.kl_coef

                    loss = 0.5 * (pg_loss / gradient_accumulation + pg_loss_rollout / gradient_accumulation)
                    
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics

    def update_policy_diffusion(self, data: DataProto) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = 0.0
        select_keys = ["old_log_probs", "advantages", "latents", "next_latents", "kl", "timesteps",
                       'prompt_embeds', 'pooled_prompt_embeds', 'negative_prompt_embeds', 'negative_pooled_prompt_embeds']
        select_keys.extend(["ref_log_probs", "ref_prev_sample_mean"])
        non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    response_mask = torch.ones_like(old_log_probs)

                    # all return: (bsz, response_length)
                    log_probs, prev_sample_mean = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=prev_sample_mean,
                            ref_log_probs=model_inputs["ref_prev_sample_mean"],
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, torch.ones_like(kld), mode=self.config.loss_avg_mode)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
