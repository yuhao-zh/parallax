"""
This module contains the http server for Parallax.

It is used to recv requests from http frontend and send prompts to executor.

"""

import multiprocessing as mp
import uvicorn
import uuid
import time
import sys
import traceback
import zmq
import zmq.asyncio
import fastapi
import asyncio

from pydantic import BaseModel
from http import HTTPStatus
from typing import Optional, Dict
from contextlib import asynccontextmanager

from parallax.utils.utils import get_zmq_socket
from parallax.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str

async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {traceback}")
        sys.exit(1)

class HTTPHandler:
    def __init__(
            self,
            executor_input_ipc_name,
            executor_output_ipc_name,
        ):
        self.asyncio_tasks = set()
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.send_to_executor = get_zmq_socket(
            context, zmq.PUSH, executor_input_ipc_name, True
        )
        self.recv_from_executor = get_zmq_socket(
            context, zmq.PULL, executor_output_ipc_name, True
        )
        self.request_result = {}

    def send_requests(self, requests: Dict):
        self.send_to_executor.send_pyobj(requests)

    async def _handle_loop(self):
        """The event loop that handles returned requests"""
        while True:
            recv_dict = await self.recv_from_executor.recv_pyobj()
            print("[ty]recv result", recv_dict)
            rid = recv_dict["rid"]
            output = recv_dict["output"]
            self.request_result[rid] = output

    async def create_handle_loop(self):
        task_loop = asyncio.create_task(print_exception_wrapper(self._handle_loop))
        await task_loop

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int

def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
):
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    return fastapi.ORJSONResponse(content=error.model_dump(), status_code=error.code)

async def v1_chat_completions(raw_request: fastapi.Request):
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))
    request_id = f"chatcmpl-{uuid.uuid4()}"
    request_json["rid"] = request_id
    http_handler.request_result[request_id] = ""
    http_handler.send_requests(request_json)
    res = ""
    while True:
        try:
            res = http_handler.request_result.get(request_id)
            if (len(res) > 0):
                print("[ty]res=", res)
                break
        except:
            break

    return res

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


class ParallaxHttpServer:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.executor_input_ipc_name = args.executor_input_ipc
        self.executor_output_ipc_name = args.executor_output_ipc

    async def run_uvicorn(self):
        config = uvicorn.Config(app,
                                host=self.host,
                                port=self.port,
                                timeout_keep_alive=5,
                                loop="uvloop",
                            )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_tasks(self):
        await asyncio.gather(self.run_uvicorn(), http_handler.create_handle_loop())

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        global http_handler
        http_handler = HTTPHandler(
            self.executor_input_ipc_name,
            self.executor_output_ipc_name,
        )

        asyncio.run(self.run_tasks())

def launch_http_server(args):
    http_server = ParallaxHttpServer(args)
    mp.Process(target=http_server.run).start()
