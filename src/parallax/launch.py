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
