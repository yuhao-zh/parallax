"""
A minimum implementation of Request objects for managing inference requests.

Each time a client initialize a generation request by sending prompt texts to the First Peer,
    it will initialize an `InitialRequest` instance.
Each Intermediate Peer will only hold `IntermediateRequest`,
    which contains part of the info of `InitialRequest`.
        * request id,
        * current position.
        * relevant hidden_states (prompt_len, hidden_size) or (1, hidden_size)
    We will pack `IntermediateRequest` and send to the following peers.

In our current design, the Last Peer won't hold `InitialRequest` for several reasons:
    * in case of weight tying, it doens't need to load and compute lm_head GEMM;
    * easier management for Request state updates (only the First Peer will handle it);
    * no need to hold tokenizer so that we can decokenize first and then stream to Client;
    * we need to announce to the First Peer to evict the request from its running batch anyway.
    - slight delay for client to recieve the streaming token.

Forming requests to batches (see scheduler.py for details):
    Our scheduler can manage both `InitialRequest` and `IntermediateRequest`.

A complete workflow:
Prefill:
    First Peer:
        * Initialize `InitialRequest`;
        (* Put the request into the schduler's pool;)
        * Feed `input_ids` into executor;
        * Store each layer's hidden states to paged KV Cache;
        * Generates the last layer (that it holds)'s hidden states;
        * Build `IntermediateRequest` with the request id,
            current position, and hidden states (prompt_len, hidden_size);
        * Sends `IntermediateRequest` to the next Peer.
    Intermediate Peers:
        * Accepts `IntermediateRequest`;
        * Feed `hidden_states` (prompt_len, hidden_size) to executor, updates caches
        * Update `IntermediateRequest` with the hidden states and send it to the next peer;
    Last Peer:
        * Accepts `IntermediateRequest`;
        * Generates and sends the first `output_id` (without weight tying) to the first Peer.

    Now, the first peer will
        1. Detokenize the request and stream it to the client;
        2. Switch request status from Prefill to Decode
        3. Add output token id to the ouputs and increase the processed length.
        4. Check if the request should be switch to Finished state, evict if necessary.

Decode:
    First Peer: Feed singleton `output_id` to executor,
        generates and sends `hidden_states` (1, hidden_size);
    Intermediate Peers: takes in `hidden_states`,
        generates and sends `hidden_states` (1, hidden_size);
    Last Peer: takes in `hidden_states`,
        generates and sends `output_id` to the first Peer.

TODO:
    1. For now we only supports Greedy Decoding,
        we need to add sampling sampling params and passing logits;
    2. Add support for multiple output_ids in a single step (e.g. beam width, top-k sampling, etc.);
    3. Accepts more generation configs like repetition penalties.
"""

import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from parallax.logging_config import get_logger

logger = get_logger(__name__)


class RequestStatus(Enum):
    """Enumeration of possible request statuses for the First Peer."""

    PENDING = "PENDING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED_EOS = "FINISHED_EOS"
    FINISHED_MAX_LENGTH = "FINISHED_MAX_LENGTH"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"


class InitialRequest:
    """
    Represents the full state of a user's generation request, managed by the First Peer.
    """

    def __init__(
        self,
        *input_ids: List[int],
        eos_token_id: int,
        request_id: Optional[str] = None,
        output_ids: Optional[List[int]] = None,
        max_new_tokens: int = 512,
        max_total_length: int = 1024,
        status: RequestStatus = RequestStatus.PENDING,
    ):
        self.request_id: str = request_id or str(uuid.uuid4())
        if not input_ids:
            raise ValueError("input_ids (prompt) cannot be empty.")
        self.input_ids = input_ids

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")
        self.max_new_tokens = max_new_tokens
        if max_total_length < max_new_tokens + len(input_ids):
            raise ValueError(
                "max_total_length must be at least max_new_tokens + length of input_ids."
            )
        self.max_total_length = max_total_length
        self.eos_token_id = eos_token_id
        self.output_ids = output_ids or []

        self.status = status
        if len(self.output_ids) > 0 and self.status == RequestStatus.PREFILLING:
            raise ValueError(f"Cannot initialize with output_ids given {self.status}.")

    @property
    def prompt_length(self) -> int:
        """Length of the input prompt (input_ids)."""
        return len(self.input_ids)

    @property
    def output_length(self) -> int:
        """Length of the generated output (output_ids)."""
        return len(self.output_ids)

    @property
    def total_length(self) -> int:
        """Total length of the sequence (input + output)."""
        return self.prompt_length + self.output_length

    @property
    def is_finished(self) -> bool:
        """Checks if the request has finished processing."""
        return self.status in [
            RequestStatus.FINISHED_EOS,
            RequestStatus.FINISHED_MAX_LENGTH,
            RequestStatus.ERROR,
            RequestStatus.CANCELLED,
        ]

    def get_model_input_for_first_peer(self) -> List[int]:
        """
        Returns the token IDs the First Peer's model should process for the current step.
        """
        if not self.output_ids:
            return self.input_ids
        return [self.output_ids[-1]]

    def commit_new_token(self, token_id: int):
        """
        Called by the First Peer when a new token is received from the Last Peer.
        Appends the token, updates length, and checks finishing conditions.
        """
        if self.is_finished:
            logger.warning(
                f"Request {self.request_id}: Attempted to commit token to a finished request."
            )
            return

        self.output_ids.append(token_id)

        # Check finishing conditions
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            self.status = RequestStatus.FINISHED_EOS
            logger.info(f"Request {self.request_id} finished: EOS token received.")
        elif self.output_length >= self.max_new_tokens:
            self.status = RequestStatus.FINISHED_MAX_LENGTH
            logger.info(f"Request {self.request_id} finished: Max new tokens generated.")
        elif self.total_length >= self.max_total_length:
            self.status = RequestStatus.FINISHED_MAX_LENGTH
            logger.info(f"Request {self.request_id} finished: Max sequence length reached.")
        else:
            if self.status == RequestStatus.PREFILLING:
                self.status = RequestStatus.DECODING


@dataclass
class IntermediateRequest:
    """
    Lightweight data packet sent between intermediate peers in the pipeline.
    This is what gets packed and sent over the network.
    """

    request_id: str
    # Position of the *last token* for which these hidden_states were generated
    current_position: int

    # Hidden states from the previous peer's computation.
    # Shape:
    #   prefill: (prompt_len, hidden_dim)
    #   decode: (1, hidden_dim)
    hidden_states: np.ndarray

    # TODO: add attention_mask, logits...
