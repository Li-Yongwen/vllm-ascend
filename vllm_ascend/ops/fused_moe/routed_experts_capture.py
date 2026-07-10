import torch
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    _file_lock,
)


class AscendRoutedExpertsCapturer(RoutedExpertsCapturer):
    """
    Capturer for routed experts with device and optional shared memory buffer.

    In EP setups each TP(EP) rank has a *different* slot_mapping where only
    its own tokens have valid (non-negative) slots.  Every rank saves its
    own valid-slot data to the same shared memory so that all slots are
    eventually covered.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_size = get_dp_group().world_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        In the Ascend EP implementation, gating runs *before* EP dispatch,
        so ``topk_ids`` contains all tokens in the batch regardless of which
        EP rank we are on.  We simply store it into ``device_buffer``.
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        n = topk_ids.shape[0]
        if layer_id >= self._device_buffer.shape[1]:
            return

        self._device_buffer[:n, layer_id, :] = topk_ids

    def save_captured_experts(
        self,
        indices,  # np.ndarray
        token_positions=None,  # np.ndarray | None
    ) -> None:
        """Save captured experts from device buffer to shared memory.

        Each TP(EP) rank writes only the slots whose index is >= 0 in its
        ``cpu_slot_mapping``.  Because different EP ranks have different
        valid slots, every slot is written by exactly one rank, and the
        shared memory ends up with complete data.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        num_tokens = len(indices)

        import sys
        print(f"[SAVE-ENTRY] tp_rank={self.tp_rank} num_tokens={num_tokens} "
              f"indices[:3]={indices[:3]}", file=sys.stderr, flush=True)

        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()

        host_indices = indices

        # Skip slots with -1 (tokens that belong to other EP ranks).
        valid_mask = host_indices >= 0
        valid_indices = host_indices[valid_mask]
        valid_data = data[valid_mask]

        if len(valid_indices) == 0:
            return

        import logging
        logger = logging.getLogger(__name__)
        logger.debug(
            "[routed_experts] save: tp_rank=%d num_tokens=%d valid=%d "
            "host_indices[:3]=%s data_nonzero=%s",
            self.tp_rank, num_tokens, len(valid_indices),
            valid_indices[:3], (valid_data != 0).any(),
        )

        with _file_lock(self._lock_file):
            self._host_buffer_view[valid_indices, :, :] = valid_data
