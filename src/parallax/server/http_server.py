"""
This module contains the http server for Parallax.

It is used to recv requests from http frontend and send prompts to executor.

"""

import multiprocessing as mp
import uvicorn
import zmq
import fastapi
from contextlib import asynccontextmanager
from sglang.srt.openai_api.protocol import (
    ChatCompletionRequest,
)
from sglang.srt.openai_api.adapter import create_error_response

from parallax.utils.utils import get_zmq_socket


async def v1_chat_completions(raw_request: fastapi.Request):
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))
    all_requests = [ChatCompletionRequest(**request_json)]
    print(all_requests)

@asynccontextmanager
async def lifespan(fast_api_app: fastapi.FastAPI):
    yield

# Fast API
app = fastapi.FastAPI(
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: fastapi.Request):
    return await v1_chat_completions(raw_request)


def launch_http_server(args):
    """
    Launch A FastAPI server that routes requests to the executor.

    Note:
    1. The HTTP server and executor both run in the main process.
    2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
    """
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=5,
            loop="uvloop",
        )
    finally:
        pass

def __main__():
    from parallax.server.server_args import parse_args
    args = parse_args()
    launch_http_server(args)
