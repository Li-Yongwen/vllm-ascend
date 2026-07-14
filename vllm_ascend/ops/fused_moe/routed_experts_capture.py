import logging

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

logger = logging.getLogger(__name__)


class AscendRoutedExpertsCapturer(RoutedExpertsCapturer):
    """
    Capturer for routed experts with device and optional shared memory buffer.

    In the Ascend EP implementation with FlashComm1 + tid2eid, each TP
    rank computes different ``topk_ids`` because ``input_ids`` is
    sharded across ranks.  The ``capture()`` call site in
    ``w8a8_dynamic.apply()`` passes **logical** expert IDs (before
    tid2eid remapping) which are identical across all ranks, so only
    TP0 needs to write to shared memory.

    NOTE: ``capture()`` may be called inside ACL graph capture
    (dummy_run), so it must NOT contain collective communication
    operations like all-gather.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_size = get_dp_group().world_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Stores the local ``topk_ids`` into ``device_buffer``.
        No collective communication is performed here because this
        method may be called inside ACL graph capture.
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

        Because ``capture()`` is called with logical expert IDs
        (identical across all TP ranks), all ranks have the same
        ``device_buffer`` data.  Only TP0 writes to shared memory.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        # All ranks have the same data; only TP0 writes to shared memory.
        if self.tp_rank != 0:
            return

        num_tokens = len(indices)
        data = self._device_buffer[:num_tokens, :, :].cpu().numpy()

        with _file_lock(self._lock_file):
            valid_mask = indices >= 0
            valid_indices = indices[valid_mask]
            valid_data = data[valid_mask]

            if len(valid_indices) > 0:
                self._host_buffer_view[valid_indices, :, :] = valid_data

            if hasattr(self, '_slot_mapping_view') and self._slot_mapping_view is not None:
                self._slot_mapping_view[:num_tokens] = indices

            if (hasattr(self, '_token_counts_view')
                    and self._token_counts_view is not None
                    and token_counts_per_req is not None
                    and num_reqs > 0):
                self._token_counts_view[:num_reqs] = token_counts_per_req[:num_reqs]

        logger.debug(
            "[SAVE] tp_rank=%d num_tokens=%d valid=%d",
            self.tp_rank,
            num_tokens,
            len(valid_indices) if len(valid_indices) > 0 else 0,
        )
