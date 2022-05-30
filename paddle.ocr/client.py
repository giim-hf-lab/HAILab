#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__all__ = []

import asyncio
import sys
from pathlib import Path

import cv2
import msgpack
from aiohttp import ClientSession


async def _main() -> int:
    images = [
        cv2.imread(str(path))
        for path in Path("data").iterdir()
    ]

    async with ClientSession(raise_for_status=True) as session:
        h, w, c = images[0].shape
        async with session.post("http://localhost:8080/single", data=msgpack.packb({
            "parameters": {
                "min_length": 2
            },
            "image": {
                "data": images[0].tobytes(),
                "height": h,
                "width": w,
                "channels": c
            }
        }), headers={
            "Content-Type": "application/json"
        }) as response:
            print(msgpack.unpackb(await response.read()))

        async with session.ws_connect("http://localhost:8080/streaming") as ws:
            await ws.send_bytes(msgpack.packb({
                "min_length": 2
            }))
            for image in images:
                h, w, c = image.shape
                await ws.send_bytes(msgpack.packb({
                    "data": image.tobytes(),
                    "height": h,
                    "width": w,
                    "channels": c
                }))
                print(msgpack.unpackb(await ws.receive_bytes()))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
