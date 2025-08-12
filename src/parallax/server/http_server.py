"""
This module contains the http server for Parallax.

It is used to recv requests from http frontend and send prompts to executor.

"""

import multiprocessing as mp
import uvicorn
import zmq
import fastapi
import logger

from pydantic import BaseModel
from http import HTTPStatus
from typing import Optional, Dict
from contextlib import asynccontextmanager

from parallax.utils.utils import get_zmq_socket


class HTTPCommunicator:
    def __init__(self, args):
        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.send_to_executor = get_zmq_socket(
            context, zmq.PUSH, args.executor_input_ipc, True
        )
        self.recv_from_executor = get_zmq_socket(
            context, zmq.PULL, args.executor_output_ipc, True
        )
    
    def send_requests(self, requests: Dict):
        self.send_to_executor.send_pyobj(requests)
        
    def recv_prompts(self):
        msgs = []
        while True:
            try:
                recv_msg = self.recv_from_executor.recv_pyobj(zmq.NOBLOCK)
                msgs.append(recv_msg)
            except zmq.ZMQError:
                break
            except Exception as e:
                logger.exception(f"Error receiving http request: {e}")
        return msgs

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

def _create_error_response(e):
    return fastapi.ORJSONResponse(
        {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
    )


async def v1_chat_completions(raw_request: fastapi.Request):
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))
    http_comm.send_requests(request_json)

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

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        global http_comm 
        http_comm = HTTPCommunicator(args)

        try:
            uvicorn.run(
                app,
                host=self.host,
                port=self.port,
                timeout_keep_alive=5,
                loop="uvloop",
            )
        finally:
            pass

def launch_http_server(args):
    http_server = ParallaxHttpServer(args)
    mp.Process(target=http_server.run).start()

if __name__ == "__main__":
    from parallax.server.server_args import parse_args
    args = parse_args()
    launch_http_server(args)
