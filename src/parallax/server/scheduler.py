"""
Scheduling requests to form batches.

A scheduler will maintain a Priority Queue for request waiting pool.
We support continuous batching, and similar to TensorRT-LLM,
    we favors prefill requests over decode requests.
"""

import heapq
import time
from typing import Dict, List, Literal, Tuple

from parallax.server.request import Request, RequestStatus
from parallax.utils.logging_config import get_logger

logger = get_logger(__name__)


class Scheduler:
    """
    A simple scheduler to manage requests and form them into batches.
    This scheduler is designed to handle requests in a FIFO manner.
    """

    def __init__(
        self,
        pad_token_id: int,
        max_batch_size: int = 16,
        max_num_tokens: int = 1024,
        prefill_priority: Literal[0, 1] = 0,
        scheduler_wait_ms: int = 500,
        micro_batch_ratio: int = 2,
    ):
        """
        Args:
            pad_token_id: The ID of the padding token used in the model;
            max_batch_size: Maximum number of running requests;
            max_num_tokens: Maxmimum number of prefill + decode tokens in a single batch;
            prefill_priority: Priority for prefill requests,
                default 0 for prefill, 1 for decode, 0 for higher priority;
            scheduler_wait_ms: The minimum time to wait before dispatching a batch;
            micro_batch_ratio: micro_batch_size = max_batch_size // micro_batch_ratio
        """
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        self.micro_batch_size = self.max_batch_size // micro_batch_ratio
        self.scheduler_wait_ms = scheduler_wait_ms
        self.pad_token_id = pad_token_id

        # Priority queue: (priority, arrival_time, request_id, request_object)
        self._request_queue: List[Tuple[int, float, str, Request]] = []
        # Keeps track of all in-flight requests
        self._running_requests: Dict[str, Request] = {}
        self._inflight_tokens: int = 0

        self.priority_map = {
            RequestStatus.PREFILLING: prefill_priority,
            RequestStatus.DECODING: 1 - prefill_priority,
        }
        self._last_dispatch_ts = time.time()
        logger.info(
            f"Scheduler initialized: max_batch_size={self.max_batch_size}, "
            f"max_num_tokens={self.max_num_tokens}, pad_token_id={self.pad_token_id}"
        )

    @property
    def num_queued_requests(self) -> int:
        """Get the number of requests in the scheduler."""
        return len(self._request_queue)

    @property
    def num_running_requests(self) -> int:
        """Get the number of requests currently being processed."""
        return len(self._running_requests)

    @property
    def has_pending_requests(self) -> bool:
        """Check if there are any pending requests in the scheduler."""
        return len(self._request_queue) > 0

    def enque_request(self, request: Request):
        """Add a request to the scheduler."""
        if request.is_finished:
            logger.warning(
                f"Request {request.request_id} is already "
                f"{request.status}. Not adding to the scheduler."
            )
            return
        arrival_time = time.time()
        priority = self.priority_map.get(request.status, 1)
        heapq.heappush(self._request_queue, (priority, arrival_time, request.request_id, request))
        logger.debug(f"Request {request.request_id} added to the scheduler.")

    def should_dispatch(self) -> bool:
        """Helper check if the scheduler should dispatch a batch."""
        waited = (time.time() - self._last_dispatch_ts) * 1000 >= self.scheduler_wait_ms
        queued = self.num_queued_requests >= self.micro_batch_size
        return waited or queued

    def form_batch(self) -> List[Request]:
        """Get the next batch of requests.

        At-most `micro_batch_size` requests will be returned.
        """
        if not self.has_pending_requests:
            logger.warning("No pending requests in the scheduler.")
            return []

        batch = []
        while True:
            if not self.has_pending_requests:
                break
            if len(batch) >= self.micro_batch_size:
                break
            _, _, rid, req = self._request_queue[0]
            cost = req.prompt_len if req.is_prefill else 1
            if cost + self._inflight_tokens > self.max_num_tokens:
                break

            heapq.heappop(self._request_queue)
            batch.append(req)
            if rid not in self._running_requests:
                self._running_requests[rid] = req
            else:
                assert req.is_decoding, "Request should be decoding if already run."
                staled_req_state = self._running_requests[rid]
                if staled_req_state.is_prefill:
                    self._inflight_tokens -= staled_req_state.prompt_len
                self._running_requests[rid] = req

            self._inflight_tokens += cost

        return batch
