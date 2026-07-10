import numpy as np
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

    In the Ascend EP implementation, gating runs *before* EP dispatch,
    so every TP(EP) rank sees the full set of tokens and produces its
    own topk_ids.  Only TP0 writes the data to shared memory using
    KV-slot-based indexing.  The scheduler reads using the same
    KV-slot calculation.
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
        token_positions=None,  # np.ndarray | None -- ignored
    ) -> None:
        """Save captured experts from device buffer to shared memory.

        Only TP0 performs the write.  Data is written using the KV-slot
        indices passed in ``indices``.  The scheduler reads using the
        same KV-slot calculation.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        # Only TP0 writes to avoid redundant / conflicting writes.
        if self.tp_rank != 0:
            return

        num_tokens = len(indices)
        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()

        import sys
        print(f"[SAVE] num_tokens={num_tokens} num_valid={np.sum(indices >= 0)} "
              f"indices[:5]={indices[:5]} data[:2,0,:3]={data[:2, 0, :3]}",
              file=sys.stderr, flush=True)

        # Write to shared memory using KV-slot indices.
        # Filter out -1 entries (padding / tokens without KV slots).
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_data = data[valid_mask]

        if len(valid_indices) > 0:
            with _file_lock(self._lock_file):
                self._host_buffer_view[valid_indices, :, :] = valid_data
