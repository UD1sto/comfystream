import asyncio
import argparse
import os
import json
import logging

from twilio.rest import Client
from aiohttp import web
from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCConfiguration,
    RTCIceServer,
    MediaStreamTrack,
)
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.codecs import h264
from pipeline import OneWayPipeline
from utils import patch_loop_datagram

logger = logging.getLogger(__name__)
logging.getLogger('aiortc.rtcrtpsender').setLevel(logging.WARNING)
logging.getLogger('aiortc.rtcrtpreceiver').setLevel(logging.WARNING)

MAX_BITRATE = 2000000
MIN_BITRATE = 2000000

class OneWayVideoStreamTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    async def recv(self):
        video = await self.pipeline()
        return video

def force_codec(pc, sender, forced_codec):
    kind = forced_codec.split("/")[0]
    codecs = RTCRtpSender.getCapabilities(kind).codecs
    transceiver = next(t for t in pc.getTransceivers() if t.sender == sender)
    codecPrefs = [codec for codec in codecs if codec.mimeType == forced_codec]
    transceiver.setCodecPreferences(codecPrefs)
    
def get_twilio_token():
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")

    if account_sid is None or auth_token is None:
        return None

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token


def get_ice_servers():
    ice_servers = []

    token = get_twilio_token()
    if token is not None:
        # Use Twilio TURN servers
        for server in token.ice_servers:
            if server["url"].startswith("turn:"):
                turn = RTCIceServer(
                    urls=[server["urls"]],
                    credential=server["credential"],
                    username=server["username"],
                )
                ice_servers.append(turn)

    return ice_servers

async def offer(request):
    pipeline = request.app["pipeline"]
    pcs = request.app["pcs"]

    params = await request.json()
    pipeline.set_prompt(params["prompt"])
    await pipeline.warm()

    offer_params = params["offer"]
    offer = RTCSessionDescription(sdp=offer_params["sdp"], type=offer_params["type"])

    ice_servers = get_ice_servers()
    pc = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=ice_servers) if ice_servers else None
    )
    pcs.add(pc)

    # Add video track immediately (server -> client)
    video_track = OneWayVideoStreamTrack(pipeline)
    sender = pc.addTrack(video_track)
    force_codec(pc, sender, "video/H264")

    # Maintain control channel functionality
    @pc.on("datachannel")
    def on_datachannel(channel):
        if channel.label == "control":
            @channel.on("message")
            async def on_message(message):
                try:
                    params = json.loads(message)
                    if params.get("type") == "get_nodes":
                        nodes_info = await pipeline.get_nodes_info()
                        response = {
                            "type": "nodes_info",
                            "nodes": nodes_info
                        }
                        channel.send(json.dumps(response))
                    elif params.get("type") == "update_prompt":
                        pipeline.set_prompt(params["prompt"])
                        channel.send(json.dumps({"type": "prompt_updated", "success": True}))
                    else:
                        logger.warning("Unhandled message type: %s", params.get("type"))
                except Exception as e:
                    logger.error("Control channel error: %s", str(e))

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state: %s", pc.connectionState)
        if pc.connectionState in ["failed", "closed"]:
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        }),
    )


async def set_prompt(request):
    pipeline = request.app["pipeline"]

    prompt = await request.json()
    logger.info("Setting prompt: %s", prompt)
    pipeline.set_prompt(prompt)

    return web.Response(content_type="application/json", text="OK")


def health(_):
    return web.Response(content_type="application/json", text="OK")


async def on_startup(app: web.Application):
    if app["media_ports"]:
        patch_loop_datagram(app["media_ports"])

    app["pipeline"] = OneWayPipeline(
        cwd=app["workspace"], disable_cuda_malloc=True, gpu_only=True
    )
    app["pcs"] = set()

async def on_shutdown(app: web.Application):
    pcs = app["pcs"]
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comfystream server")
    parser.add_argument("--port", default=8888, help="Set the signaling port")
    parser.add_argument(
        "--media-ports", default=None, help="Set the UDP ports for WebRTC media"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Set the host")
    parser.add_argument(
        "--workspace", default=None, required=True, help="Set Comfy workspace"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    app = web.Application()
    app["media_ports"] = args.media_ports.split(",") if args.media_ports else None
    app["workspace"] = args.workspace

    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_post("/offer", offer)
    app.router.add_post("/prompt", set_prompt)
    app.router.add_get("/", health)

    web.run_app(app, host=args.host, port=int(args.port))