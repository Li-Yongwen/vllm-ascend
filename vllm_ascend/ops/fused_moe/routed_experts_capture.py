import math

import torch
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    _file_lock,
)


class AscendRoutedExpertsCapturer(RoutedExpertsCapturer):
    """
    Capturer for routed experts with device and optional shared memory buffer.

    In EP setups all TP(EP) ranks see the same global topk_ids after
    all_gather.  Only TP0 performs the actual save to shared memory
    (the others share the same dp_rank == 0 buffer but skip the write
    to avoid redundant work).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_size = get_dp_group().world_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        When EP is active, we all_gather the topk_ids so that every
        rank's device_buffer contains the full set of routing decisions.
        This allows save_captured_experts to correctly pair data with
        the global slot_mapping.
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]

            if n == total:
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif self.tp_size > 1 or self.dp_size > 1:
                token_num_per_tp = math.ceil(token_num_per_dp / self.tp_size)
                pad_size = token_num_per_tp - topk_ids.shape[0]
                if pad_size > 0:
                    topk_ids = torch.nn.functional.pad(topk_ids, (0, 0, 0, pad_size))
                full_topk_ids = tensor_model_parallel_all_gather(topk_ids, dim=0)
                topk_ids = full_topk_ids[:token_num_per_dp]
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                raise AssertionError(
                    "AscendRoutedExpertsCapturer: unexpected topk_ids batch dim "
                    f"{n} (expected {total} or {token_num_per_dp} "
                    f"for dp_rank={self.dp_rank})"
                )

        if layer_id >= self._device_buffer.shape[1]:
            return

        self._device_buffer[:token_num_per_dp, layer_id, :] = topk_ids[
            start_loc:end_loc, :
        ]

    def save_captured_experts(
        self,
        indices,  # np.ndarray
        token_positions=None,  # np.ndarray | None
    ) -> None:
        """Save captured experts from device buffer to shared memory.

        After all_gather in capture(), device_buffer contains the full
        DP-rank view.  We use the global slot_mapping (indices) to write
        the data.  Only TP0 performs the actual write; other TP ranks
        have the same data but skip to avoid redundant I/O.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        # In EP setups all ranks have identical device_buffer content
        # (thanks to all_gather) and identical slot_mapping.  Let only
        # TP0 do the write to shared memory to avoid redundant work.
        if self.tp_rank != 0:
            return

        num_tokens = len(indices)

        import sys
        print(f"[SAVE-ENTRY] tp_rank={self.tp_rank} num_tokens={num_tokens} "
              f"indices[:3]={indices[:3]}", file=sys.stderr, flush=True)

        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()

        host_indices = indices

        # Skip slots with -1 (padding tokens that have no KV cache slot).
        valid_mask = host_indices >= 0
        valid_indices = host_indices[valid_mask]
        valid_data = data[valid_mask]

        if len(valid_indices) == 0:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "[routed_experts] save: tp_rank=%d num_tokens=%d valid=%d "
            "indices[:3]=%s host_indices[:3]=%s data_nonzero=%s",
            self.tp_rank, num_tokens, len(valid_indices),
            indices[:3], valid_indices[:3], (valid_data != 0).any(),
        )

        with _file_lock(self._lock_file):
            self._host_buffer_view[valid_indices, :, :] = valid_data
