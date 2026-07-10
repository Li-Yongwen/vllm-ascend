import torch
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

    In EP setups each TP(EP) rank stores only the tokens it is responsible
    for into ``device_buffer`` and saves them to the same shared memory
    (dp_rank == 0).  The scheduler-side ``RoutedExpertsReader`` reads from
    that single shared memory which is populated by all TP ranks.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_size = get_dp_group().world_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        Only the tokens visible to the current TP rank are stored.
        No all_gather is performed — each rank independently saves its
        own slice, and the shared memory is populated by all ranks.
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

        In EP setups all TP(EP) ranks write to the same shared memory
        (dp_rank == 0).  Each rank saves only the tokens it captured.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        num_tokens = len(indices)
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
        logger.debug(
            "[routed_experts] save: tp_rank=%d num_tokens=%d valid=%d "
            "indices[:3]=%s host_indices[:3]=%s data_nonzero=%s",
            self.tp_rank, num_tokens, len(valid_indices),
            indices[:3], valid_indices[:3], (valid_data != 0).any(),
        )

        with _file_lock(self._lock_file):
            self._host_buffer_view[valid_indices, :, :] = valid_data
