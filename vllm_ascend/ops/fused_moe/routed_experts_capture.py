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
    KV-slot-based indexing.  The scheduler reads the slot_mapping from
    shared memory (written by the worker) to locate each token's data.
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
        num_reqs=0,  # number of requests in this step
        token_counts_per_req=None,  # np.ndarray | None -- token count per request
    ) -> None:
        """Save captured experts from device buffer to shared memory.

        Only TP0 performs the write.  Data is written using the KV-slot
        indices passed in ``indices``.  The slot_mapping and token counts
        are also written to shared memory so the scheduler can locate
        each token's data without computing slot indices itself.
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

        # Debug: count non-zero tokens in data
        import sys
        non_zero_count = int(np.any(data != 0, axis=(1, 2)).sum())
        print(f"[SAVE] tp_rank={self.tp_rank} num_tokens={num_tokens} "
              f"non_zero_data_tokens={non_zero_count} "
              f"indices_min={indices.min()} indices_max={indices.max()}",
              file=sys.stderr, flush=True)

        with _file_lock(self._lock_file):
            # Write routed_experts data using KV-slot indices.
            # Filter out -1 entries (padding / tokens without KV slots).
            valid_mask = indices >= 0
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]

            if len(valid_indices) > 0:
                self._host_buffer_view[valid_indices, :, :] = valid_data

            # Write slot_mapping to shared memory for scheduler to read.
            if hasattr(self, '_slot_mapping_view') and self._slot_mapping_view is not None:
                self._slot_mapping_view[:num_tokens] = indices

            # Write token counts per request for scheduler to locate tokens.
            if (hasattr(self, '_token_counts_view')
                    and self._token_counts_view is not None
                    and token_counts_per_req is not None
                    and num_reqs > 0):
                self._token_counts_view[:num_reqs] = token_counts_per_req[:num_reqs]
