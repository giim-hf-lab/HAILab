# -*- coding: utf-8 -*-
__all__ = ["server"]

import logging
from typing import Any, Callable, Coroutine, List

import numpy as np
import msgpack
from fastapi import FastAPI, Request, Response, status
from fastapi.routing import APIRoute
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError, conbytes, conint, constr

from ._infer import Inference

_logger = logging.getLogger("server")


class CustomRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = msgpack.unpackb(content) if (content := await self.body()) else None
        return self._json


class CustomRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        return self._custom_route_handler

    async def _custom_route_handler(self, request: Request) -> Response:
        return await APIRoute.get_route_handler(self)(CustomRequest(request.scope, request.receive))


class CustomResponse(Response):
    def render(self, content: Any) -> bytes:
        return b"" if content is None else msgpack.packb(content)


server = FastAPI(openapi_url=None, default_response_class=CustomResponse)
server.router.route_class = CustomRoute


class InvalidImage(ValueError):
    def __init__(self, length: int, height: int, width: int, channels: int) -> None:
        ValueError.__init__(self, f"image data (length {length}) cannot be reshaped to {width}x{height}x{channels}")


@server.exception_handler(Exception)
async def _unexpected(_: Request, e: Exception) -> Response:
    _logger.error("Unexpected error.", exc_info=e)
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@server.exception_handler(InvalidImage)
async def _unprocessable(_: Request, e: InvalidImage) -> Response:
    _logger.error("Unprocessable image.", exc_info=e)
    return Response(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


class InferenceParameters(BaseModel):
    min_length: conint(strict=True, gt=0) = 1
    top_k: conint(strict=True, gt=0) = None
    keywords: List[constr(strict=True, min_length=1)] = []
    exclude: bool = True


class Image(BaseModel):
    data: conbytes(strict=True, min_length=1)
    height: conint(strict=True, gt=0)
    width: conint(strict=True, gt=0)
    channels: conint(strict=True, gt=0)


@server.post("/single")
def _single(parameters: InferenceParameters, image: Image):
    try:
        image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, image.channels)
    except ValueError as e:
        raise InvalidImage(len(image.data), image.height, image.width, image.channels) from e
    return Inference(parameters.min_length, parameters.top_k, parameters.keywords, parameters.exclude)(image)


@server.websocket("/streaming")
async def _streaming(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        parameters = InferenceParameters.parse_obj(msgpack.unpackb(await websocket.receive_bytes()))
    except ValidationError as e:
        _logger.error("Invalid parameters.", exc_info=e)
        await websocket.close(status.HTTP_422_UNPROCESSABLE_ENTITY)
        return

    while True:
        try:
            try:
                image = Image.parse_obj(msgpack.unpackb(await websocket.receive_bytes()))
            except ValidationError as e:
                _logger.error("Invalid image.", exc_info=e)
                await websocket.close(status.HTTP_422_UNPROCESSABLE_ENTITY)
                return
            except WebSocketDisconnect as e:
                _logger.info("Client disconnected.", exc_info=e)
                return

            try:
                image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, image.channels)
            except ValueError as e:
                _logger.error("Invalid image.", exc_info=e)
                await websocket.close(status.HTTP_422_UNPROCESSABLE_ENTITY)
                return

            result = Inference(parameters.min_length, parameters.top_k, parameters.keywords, parameters.exclude)(image)
            try:
                await websocket.send_bytes(msgpack.packb(result))
            except WebSocketDisconnect as e:
                _logger.info("Client disconnected.", exc_info=e)
                return
        except Exception as e:
            _logger.error("Unexpected error.", exc_info=e)
            await websocket.close(status.HTTP_500_INTERNAL_SERVER_ERROR)
            return
