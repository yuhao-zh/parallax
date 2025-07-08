"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the P2P server and the executor.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B-MLX-bf16 \
    --kv-max-tokens-in-cache 1000000 \
    --max-num-tokens-in-batch 16384 \
    --max-batch-size 128 \
    --start-layer 14 \
    --end-layer 28 \
    --initial-peers {peer of GPU which hold the first half model}
"""

import multiprocessing
import tempfile

from parallax.p2p.server import launch_p2p_server
from parallax.server.executor import Executor
from parallax.server.server_args import parse_args
from parallax.utils.logging_config import get_logger

logger = get_logger("parallax.launch")

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    try:
        args = parse_args()

        args.recv_from_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"
        args.send_to_peer_addr = f"ipc://{tempfile.NamedTemporaryFile().name}"

        logger.info(f"recv_from_peer_addr: {args.recv_from_peer_addr}")
        logger.info(f"send_to_peer_addr: {args.send_to_peer_addr}")

        launch_p2p_server(args)

        executor = Executor.create_from_args(args)
        try:
            executor.run_loop()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        finally:
            executor.shutdown()

    except Exception as e:
        logger.exception(e)
