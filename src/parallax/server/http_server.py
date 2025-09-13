# pylint: disable=no-else-return
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

import asyncio
import json
import multiprocessing as mp
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, Optional

import aiohttp
import fastapi
import uvicorn
import zmq
import zmq.asyncio
from fastapi.responses import ORJSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.datastructures import State

from parallax.utils.logging_config import get_logger
from parallax.utils.utils import get_zmq_socket

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
        error_trace = get_exception_traceback()
        logger.error(f"TokenizerManager hit an exception: {error_trace}")
        sys.exit(1)


@dataclass
class HTTPRequestInfo:
    """HTTP Request information"""

    id: str
    text: str = ""
    stream: bool = False
    finish_reason: str = None
    object: str = "chat.completion"
    model: str = "default"
    create_time: float = 0.0
    update_time: float = 0.0
    logprobs: float = None
    matched_stop: int = None
    # usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # helper
    is_finish: bool = False
    stream_offset: int = 0


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
        self.send_to_executor = get_zmq_socket(context, zmq.PUSH, executor_input_ipc_name, True)
        self.recv_from_executor = get_zmq_socket(context, zmq.PULL, executor_output_ipc_name, True)
        self.processing_requests: Dict[str, HTTPRequestInfo] = {}

    def create_request(self, request: Dict):
        """Creates a new request information"""
        rid = request["rid"]
        stream = request.get("stream", False)
        model = request.get("model", "default")
        chat_object = "chat.completion.chunk" if stream else "chat.completion"
        create_time = time.time()
        update_time = create_time
        request_info = HTTPRequestInfo(
            id=rid,
            stream=stream,
            model=model,
            object=chat_object,
            create_time=create_time,
            update_time=update_time,
        )
        self.processing_requests[rid] = request_info

    def release_request(self, rid: str):
        """Releases the request resources"""
        del self.processing_requests[rid]

    def send_request(self, request: Dict):
        """Sends the request to model executor using IPC."""
        self.send_to_executor.send_pyobj(request)

    def _generate_stream_helper(self, rid, is_first, is_last):
        """generate_stream_response helper function"""
        request_info = self.processing_requests[rid]
        if is_first:
            role = "assistant"
            content = ""
        elif is_last:
            content = None
            role = None
        else:
            text_length = len(request_info.text)
            content = request_info.text[request_info.stream_offset :]
            request_info.stream_offset = text_length
            role = None
        response = {
            "id": rid,
            "object": request_info.object,
            "model": request_info.model,
            "created": request_info.create_time,
            "choices": [
                {
                    "index": 0,
                    "logprobs": request_info.logprobs,
                    "finish_reason": request_info.finish_reason,
                    "matched_stop": request_info.matched_stop,
                },
            ],
            "usage": None,
        }
        choice = response["choices"][0]
        choice["delta"] = {
            "role": role,
            "content": content,
            "reasoning_content": None,
            "tool_calls": None,
        }
        response_json = json.dumps(response, separators=(",", ":"))
        response_str = f"data: {response_json}\n\n"
        return response_str.encode()

    def generate_stream_response(self, rid):
        """Generates a streaming response"""
        try:
            first_flag = True
            first_resposne = self._generate_stream_helper(rid, True, False)

            # Intermediate response
            while True:
                request_info = self.processing_requests[rid]
                if request_info.is_finish:
                    break
                text_length = len(request_info.text)
                if text_length == request_info.stream_offset:
                    continue
                response = self._generate_stream_helper(rid, False, False)
                if first_flag:
                    first_flag = False
                    response = first_resposne + response
                yield response

            # Finish response
            last_response = self._generate_stream_helper(rid, False, True)
            last_response = last_response + b"data: [DONE]\n\n"
            yield last_response
        except (asyncio.CancelledError, aiohttp.ClientError, Exception):
            pass

    def generate_non_stream_response(self, rid):
        """Generates a non-streaming response"""
        request_info = self.processing_requests[rid]
        response = {
            "id": rid,
            "object": request_info.object,
            "model": request_info.model,
            "created": request_info.create_time,
            "choices": [
                {
                    "index": 0,
                    "logprobs": request_info.logprobs,
                    "finish_reason": request_info.finish_reason,
                    "matched_stop": request_info.matched_stop,
                },
            ],
            "usage": {
                "prompt_tokens": request_info.prompt_tokens,
                "total_tokens": request_info.prompt_tokens + request_info.completion_tokens,
                "completion_tokens": request_info.completion_tokens,
                "prompt_tokens_details": None,
            },
        }
        choice = response["choices"][0]
        choice["messages"] = {
            "role": "assistant",
            "content": request_info.text,
            "reasoning_content": None,
            "tool_calls": None,
        }
        return response

    async def _handle_loop(self):
        """The event loop that handles returned requests"""
        while True:
            recv_dict = await self.recv_from_executor.recv_pyobj()
            rid = recv_dict["rid"]
            output = recv_dict["output"]
            if rid in self.processing_requests:
                request_info = self.processing_requests[rid]
                request_info.update_time = time.time()
                if recv_dict.get("eos", False) or output == "<|im_end|>":
                    request_info.finish_reason = "stop"
                    request_info.matched_stop = 0
                    request_info.is_finish = True
                elif recv_dict.get("length", False):
                    request_info.text += output
                    if len(output) > 0:
                        request_info.completion_tokens += 1
                    request_info.finish_reason = "length"
                    request_info.is_finish = True
                else:
                    request_info.text += output
                    if len(output) > 0:
                        request_info.completion_tokens += 1

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
    return ORJSONResponse(content=error.model_dump(), status_code=error.code)


# Fast API
app = fastapi.FastAPI(
    openapi_url="/openapi.json",
)


async def init_app_states(state: State, executor_input_ipc: str, executor_output_ipc: str):
    """Init FastAPI app states, including http handler, etc."""
    state.http_handler = HTTPHandler(
        executor_input_ipc,
        executor_output_ipc,
    )


async def v1_chat_completions(raw_request: fastapi.Request):
    """
    Handles the v1/chat/completions requests asynchronously.
    It gets the prompts from HTTPHandler and returns to the frontend.
    """
    try:
        request_json = await raw_request.json()
    except Exception as e:
        return create_error_response("Invalid request body, error: ", str(e))
    request_id = str(uuid.uuid4())
    request_json["rid"] = request_id
    app.state.http_handler.create_request(request_json)
    app.state.http_handler.send_request(request_json)
    req = app.state.http_handler.processing_requests.get(request_id)
    is_stream = req.stream

    if is_stream:
        return StreamingResponse(
            app.state.http_handler.generate_stream_response(request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        while True:
            await asyncio.sleep(0.01)
            req = app.state.http_handler.processing_requests.get(request_id)
            is_finish = req.is_finish
            if is_finish:
                break
        response = app.state.http_handler.generate_non_stream_response(request_id)
        app.state.http_handler.release_request(request_id)
        return ORJSONResponse(status_code=200, content=response)


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
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            timeout_keep_alive=5,
            loop="uvloop",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def run_tasks(self):
        """Gather results of all asyncio tasks"""
        await asyncio.gather(self.run_uvicorn(), app.state.http_handler.create_handle_loop())

    def run(self):
        """
        Launch A FastAPI server that routes requests to the executor.

        Note:
        1. The HTTP server and executor both run in the main process.
        2. Inter-process communication is done through IPC (each process uses a different port) via the ZMQ library.
        """
        asyncio.run(
            init_app_states(app.state, self.executor_input_ipc_name, self.executor_output_ipc_name)
        )
        asyncio.run(self.run_tasks())


def launch_http_server(args):
    """
    Launch function of frontend server.
    It creates a sub-process for the http server.
    """
    http_server = ParallaxHttpServer(args)
    mp.Process(target=http_server.run).start()
