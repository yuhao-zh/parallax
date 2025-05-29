"""
Server runner for Parallax, which initializes a DHT and starts serving model shards.
"""

from dataclasses import dataclass
from typing import List, Optional, Type

import hivemind
from mlx import nn
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3Block

from parallax.logging_config import get_logger
from parallax.server.executor import ModelExecutor

logger = get_logger(__name__)


@dataclass
class DHTConfig:
    """Configuration for the DHT used in the Parallax server."""

    public_ip: str
    port: int = 40000
    host_maddrs: Optional[List[str]] = None
    initial_peers: Optional[List[str]] = None
    identity_path: str = "parallax.key"


class ServerRunner:
    # pylint: disable=too-few-public-methods
    """Server runner for Parallax, which initializes a DHT and starts serving model shards."""

    def __init__(
        self,
        model_id: str,
        start_layer: int,
        end_layer: int,
        dht_config: DHTConfig,
        block_class: Type[nn.Module] = Qwen3Block,
    ):
        self.executor = ModelExecutor(model_id, start_layer, end_layer, block_class)
        self.model_shard = self.executor.model_shard

        self.port = dht_config.port
        self.public_ip = dht_config.public_ip
        if dht_config.host_maddrs is None:
            self.host_maddrs = [f"/ip4/0.0.0.0/tcp/{self.port}", f"/ip6/::/tcp/{self.port}"]
        self.announce_maddrs = [f"/ip4/{self.public_ip}/tcp/{self.port}"]
        self.initial_peers = dht_config.initial_peers or []

        logger.info("Starting DHT")

        self.dht = hivemind.DHT(
            initial_peers=self.initial_peers,
            identity_path=dht_config.identity_path,
            num_workers=self.model_shard.n_layers,
            start=True,
        )

        visible_maddrs = [str(a) for a in self.dht.get_visible_maddrs()]
        logger.info(f"Running server on {visible_maddrs}")
        logger.info("Peer-ID %s   visible at %s", self.dht.peer_id, self.dht.get_visible_maddrs())
