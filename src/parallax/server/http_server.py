"""
This module contains the http frontend server for Parallax.
Two classes that handles a post request from the frontend service:

  -- ParallaxHttpServer:
    The uvicorn server that communicates with the frontend and posts responses
    to users.
    This module launches a subprocess.

  -- HTTPHandler:
    1.Gets requests from ParallaxHttpServer and maintains status of these requests.
    2.Send raw requests by ipc to parallax executor.
    3.Waits for ipc response from the executor and stores the results.
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
    """Traceback function to handle asyncio function errors"""
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
    """
    A global handler that maintains raw requests. It has 2 main functions:
    1.Interprocess communicate with the model executor.
    2.Maintains the request -> prompts dict for ParallaxHTTPServer.
    """
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
        self.request_finish = {}

    def send_requests(self, requests: Dict):
        """Sends the request to model executor using IPC."""
        self.send_to_executor.send_pyobj(requests)

    async def _handle_loop(self):
        """The event loop that handles returned requests"""
        while True:
            recv_dict = await self.recv_from_executor.recv_pyobj()
            rid = recv_dict["rid"]
            output = recv_dict["output"]
            self.request_result[rid] += output
            if output == "<|im_end|>":
                self.request_finish[rid] = True

    async def create_handle_loop(self):
        """Create asyncio event loop task function"""
        task_loop = asyncio.create_task(print_exception_wrapper(self._handle_loop))
        await task_loop

class ErrorResponse(BaseModel):
    """An Error data structure."""
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
    """Creates a json error response for the frontend."""
    error = ErrorResponse(message=message, type=err_type, code=status_code.value)
    return fastapi.ORJSONResponse(content=error.model_dump(), status_code=error.code)

async def v1_chat_completions(raw_request: fastapi.Request):
    """"""
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))
    request_id = f"chatcmpl-{uuid.uuid4()}"
    request_json["rid"] = request_id
    http_handler.request_result[request_id] = ""
    http_handler.request_finish[request_id] = False
    http_handler.send_requests(request_json)
    res = ""
    while True:
        await asyncio.sleep(0.1)
        res = http_handler.request_result.get(request_id)
        is_finish = http_handler.request_finish.get(request_id)
        if is_finish:
            break
    del http_handler.request_result[request_id]
    del http_handler.request_finish[request_id]
    return res

@asynccontextmanager
async def lifespan(fast_api_app: fastapi.FastAPI):
    """
    A lifespan function for the uvicorn app.
    Add in-life logic and shutdown logic in the future.
    """
    yield

# Fast API
app = fastapi.FastAPI(
    lifespan=lifespan,
    openapi_url="/openapi.json",
)

@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: fastapi.Request):
    """OpenAI v1/chat/complete post function"""
    return await v1_chat_completions(raw_request)


class ParallaxHttpServer:
    """
    The frontend http server drived by asyncio.
    Since uvicorn API runs an asyncio task, we need a new uvicorn
    wrapper to run other async tasks.
    """
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.executor_input_ipc_name = args.executor_input_ipc
        self.executor_output_ipc_name = args.executor_output_ipc

    async def run_uvicorn(self):
        """
        Since uvicorn.run() uses asyncio.run, we need another wrapper
        to create a uvicorn asyncio task to run multiple tasks.
        """
        config = uvicorn.Config(app,
                                host=self.host,
                                port=self.port,
                                timeout_keep_alive=5,
                                loop="uvloop",
                            )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_tasks(self):
        """Gather results of all asyncio tasks"""
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
    """
    Launch function of frontend server.
    It creates a sub-process for the http server.
    """
    http_server = ParallaxHttpServer(args)
    mp.Process(target=http_server.run).start()
