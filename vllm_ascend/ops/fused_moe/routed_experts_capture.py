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

    In the Ascend EP implementation, each TP/EP rank may observe a
    different subset of tokens after EP dispatch.  The ``capture()``
    method all-gathers ``topk_ids`` across TP ranks along the token
    dimension so that every rank's ``device_buffer`` contains the
    complete routing data.  Similarly, ``save_captured_experts()``
    all-gathers the ``indices`` (slot_mapping) so that TP0 can write
    every token's data to shared memory.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dp_size = get_dp_group().world_size

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """
        Capture expert routing decisions for a specific layer.

        In EP mode each TP rank may have ``topk_ids`` for only a
        subset of tokens.  We all-gather along dim=0 so every rank
        has the complete routing data for all tokens in the batch.
        """
        if self._device_buffer is None:
            raise RuntimeError("Buffer not initialized. Call init_buffer() first.")

        n = topk_ids.shape[0]
        if layer_id >= self._device_buffer.shape[1]:
            return

        if self.tp_size > 1 and n > 0:
            from vllm.distributed import get_tp_group
            # All-gather along the token dimension so every rank has
            # the complete topk_ids for all tokens.
            gathered = get_tp_group().all_gather(topk_ids, dim=0)
            self._device_buffer[:gathered.shape[0], layer_id, :] = gathered
        else:
            self._device_buffer[:n, layer_id, :] = topk_ids

    def save_captured_experts(
        self,
        indices,  # np.ndarray
        token_positions=None,  # np.ndarray | None -- ignored
        num_reqs=0,  # number of requests in this step
        token_counts_per_req=None,  # np.ndarray | None -- token count per request
    ) -> None:
        """Save captured experts from device buffer to shared memory.

        After ``capture()`` all-gathers ``topk_ids``, every rank has
        the complete routing data in ``device_buffer``.  We also
        all-gather ``indices`` so that TP0 can write every token's
        data to the correct shared memory slot.
        """
        if self._lock_file is None:
            return
        if self._host_buffer_view is None:
            return
        if self._device_buffer is None:
            return

        num_tokens = len(indices)

        # All-gather indices across TP ranks so TP0 has the complete
        # slot mapping for all tokens, not just its own subset.
        if self.tp_size > 1 and num_tokens > 0:
            from vllm.distributed import get_tp_group
            tp_group = get_tp_group()
            indices_device = torch.tensor(
                indices, dtype=torch.int32, device=self._device_buffer.device
            )
            gathered_device = tp_group.all_gather(indices_device, dim=0)
            all_indices = gathered_device.cpu().numpy()
        else:
            all_indices = indices

        total_tokens = len(all_indices)
        data = self._device_buffer[:total_tokens, :, :].cpu().numpy()

        # Only TP0 writes to shared memory.
        if self.tp_rank != 0:
            return

        with _file_lock(self._lock_file):
            valid_mask = all_indices >= 0
            valid_indices = all_indices[valid_mask]
            valid_data = data[valid_mask]

            if len(valid_indices) > 0:
                self._host_buffer_view[valid_indices, :, :] = valid_data

            if hasattr(self, '_slot_mapping_view') and self._slot_mapping_view is not None:
                self._slot_mapping_view[:total_tokens] = all_indices

            if (hasattr(self, '_token_counts_view')
                    and self._token_counts_view is not None
                    and token_counts_per_req is not None
                    and num_reqs > 0):
                # All-gather token_counts_per_req across TP ranks.
                if self.tp_size > 1:
                    from vllm.distributed import get_tp_group
                    tc_t = torch.tensor(
                        token_counts_per_req, dtype=torch.int32,
                        device=self._device_buffer.device
                    )
                    gathered_tc = get_tp_group().all_gather(tc_t, dim=0)
                    all_tc = gathered_tc.cpu().numpy()
                    total_reqs = len(all_tc)
                else:
                    all_tc = token_counts_per_req
                    total_reqs = num_reqs
                self._token_counts_view[:total_reqs] = all_tc[:total_reqs]

        logger.debug(
            "[SAVE] tp_rank=%d num_tokens=%d total_tokens=%d valid=%d",
            self.tp_rank,
            num_tokens,
            total_tokens,
            len(valid_indices) if len(valid_indices) > 0 else 0,
        )
