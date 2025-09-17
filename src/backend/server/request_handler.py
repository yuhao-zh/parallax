from typing import Dict

import aiohttp
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from backend.utils.logging_config import get_logger

logger = get_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)


class RequestHandler:
    def __init__(self):
        self.scheduler_manage = None

    def set_scheduler_manage(self, scheduler_manage):
        self.scheduler_manage = scheduler_manage

    async def v1_completions(self, request_data: Dict, request_id: str, received_ts: int):
        if not self.is_schedule_success():
            return JSONResponse(
                content={
                    "error": "Server is not ready",
                },
                status_code=500,
            )

        routing_table = self.scheduler_manage.get_routing_table(request_id, received_ts)

        if routing_table is None or len(routing_table) == 0:
            return JSONResponse(
                content={
                    "error": "Routing table not found",
                },
                status_code=500,
            )

        request_data["routing_table"] = routing_table
        call_url = self.scheduler_manage.get_call_url_by_node_id(routing_table[0])

        if not call_url:
            return JSONResponse(
                content={
                    "error": "Call url not found",
                },
                status_code=500,
            )

        # call_url = "https://4ngutn4im3qjjt-3000.proxy.runpod.net"
        url = call_url + "/v1/completions"

        async def stream_completions(url, params):
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as http_session:
                async with http_session.post(url, json=params) as response:
                    logger.info(f"post: {request_id}, code: {response.status}, params: {params}")

                    if response.status != 200:
                        error_msg = f"Upstream service returned status {response.status}, response: {response}"
                        logger.error(f"completions error: {error_msg}, request_id: {request_id}")
                        raise HTTPException(status_code=response.status, detail=error_msg)

                    async for chunk in response.content:
                        if chunk:
                            yield chunk

        # chat调用
        return StreamingResponse(
            stream_completions(url, request_data),
            media_type="text/event-stream",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    async def v1_chat_completions(self, request_data: Dict, request_id: str, received_ts: int):
        if not self.is_schedule_success():
            return JSONResponse(
                content={
                    "error": "Server is not ready",
                },
                status_code=500,
            )

        routing_table = self.scheduler_manage.get_routing_table(request_id, received_ts)

        if routing_table is None or len(routing_table) == 0:
            return JSONResponse(
                content={
                    "error": "Routing table not found",
                },
                status_code=500,
            )

        request_data["routing_table"] = routing_table
        call_url = self.scheduler_manage.get_call_url_by_node_id(routing_table[0])

        if not call_url:
            return JSONResponse(
                content={
                    "error": "Call url not found",
                },
                status_code=500,
            )

        # call_url = "https://4ngutn4im3qjjt-3000.proxy.runpod.net"
        url = call_url + "/v1/chat/completions"

        async def stream_chat_completions(url, params):
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as http_session:
                async with http_session.post(url, json=params) as response:
                    logger.info(f"post: {request_id}, code: {response.status}, params: {params}")

                    if response.status != 200:
                        error_msg = f"Upstream service returned status {response.status}, response: {response}"
                        logger.error(f"completions error: {error_msg}, request_id: {request_id}")
                        raise HTTPException(status_code=response.status, detail=error_msg)

                    async for chunk in response.content:
                        if chunk:
                            yield chunk

        # chat调用
        return StreamingResponse(
            stream_chat_completions(url, request_data),
            media_type="text/event-stream",
            headers={
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "no-cache",
            },
        )

    def is_schedule_success(self):
        if self.scheduler_manage is None:
            return False

        schedule_status = self.scheduler_manage.get_schedule_status()
        if schedule_status == "success":
            return True
        else:
            return False
