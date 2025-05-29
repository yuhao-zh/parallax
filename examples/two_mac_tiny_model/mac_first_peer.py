"""Example for running first peer of MAC."""

from parallax.server.server import DHTConfig, ServerRunner

dht_config = DHTConfig(public_ip="108.211.108.182")
REPO_ID = "mlx-community/Qwen3-0.6B-bf16"
server_runner = ServerRunner(REPO_ID, 0, 12, dht_config)
