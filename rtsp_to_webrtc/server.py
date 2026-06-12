"""
Video Protocol to WebRTC Bridge Server

Receives video from various protocols (RTSP, ONVIF, GB28181, RTMP, HLS)
and serves via WebRTC for browser playback.

MQTT broker and subscriber are in mqtt_broker.py.
Uses the video_sources module for protocol abstraction.
"""

import argparse
import asyncio
import json
import logging
import os
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from video_sources import create_video_source, detect_protocol, PROTOCOL_REGISTRY
from mqtt_broker import (
    start_mqtt_broker,
    start_mqtt_subscriber,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_TOPIC,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webrtc-server")

# Global variables
DEFAULT_URL = "rtsp://admin:admin@192.168.42.1:554/live"
pcs = set()

# Get the directory where this script is located
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# HTTP/WebRTC Routes
# ============================================================================

async def index(request):
    """Serve the main HTML page."""
    html_path = os.path.join(ROOT_DIR, "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return web.Response(content_type="text/html", text=content)
    except FileNotFoundError:
        logger.error(f"index.html not found at {html_path}")
        return web.Response(
            content_type="text/html",
            text=f"<h1>Error: index.html not found</h1><p>Expected at: {html_path}</p>",
            status=500,
        )


async def protocols(request):
    """Return list of supported protocols as JSON."""
    proto_list = []
    for name, cls in PROTOCOL_REGISTRY.items():
        proto_list.append({
            "name": name,
            "class": cls.__name__,
            "description": cls.__doc__.strip().split("\n")[0] if cls.__doc__ else "",
        })
    return web.Response(
        content_type="application/json",
        text=json.dumps(proto_list),
    )


async def offer(request):
    """Handle WebRTC offer from browser and return answer."""
    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Get video source URL and optional protocol from client
    source_url = params.get("sourceUrl", params.get("rtspUrl", DEFAULT_URL))
    protocol = params.get("protocol", None)

    try:
        detected_proto = protocol or detect_protocol(source_url)
    except ValueError as e:
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            status=400,
        )

    logger.info(f"Request: protocol={detected_proto}, url={source_url}")

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info(f"Connection state: {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer_sdp)

    try:
        video_track = create_video_source(source_url, protocol=protocol)
        sender = pc.addTrack(video_track)

        # Increase VP8 bitrate to avoid blurry video
        try:
            params = sender.getParameters()
            if params and params.encodings:
                params.encodings[0].maxBitrate = 2_000_000
                sender.setParameters(params)
                logger.info("[WebRTC] VP8 maxBitrate set to 2Mbps")
        except Exception as e:
            logger.debug(f"[WebRTC] Failed to set encoding parameters: {e}")

        logger.info(f"Video track added: {video_track.__class__.__name__}")
    except (ConnectionError, Exception) as e:
        logger.error(f"Failed to create video track: {e}")
        await pc.close()
        pcs.discard(pc)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"error": str(e)}),
            status=500,
        )

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


# ============================================================================
# Application Lifecycle
# ============================================================================

async def on_startup(app):
    """Start MQTT Broker and subscriber on app startup."""
    broker = await start_mqtt_broker()
    app["mqtt_broker"] = broker

    if broker:
        await asyncio.sleep(0.5)
        mqtt_client = start_mqtt_subscriber()
        app["mqtt_client"] = mqtt_client


async def on_shutdown(app):
    """Clean up peer connections and MQTT on shutdown."""
    mqtt_client = app.get("mqtt_client")
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        logger.info("[MQTT Client] Disconnected")

    broker = app.get("mqtt_broker")
    if broker:
        await broker.shutdown()

    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("All peer connections closed")


def main():
    global DEFAULT_URL

    parser = argparse.ArgumentParser(
        description="Video Protocol to WebRTC Bridge Server (with embedded MQTT Broker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported protocols:
  rtsp     - RTSP (Real-Time Streaming Protocol)
  onvif    - ONVIF (Open Network Video Interface Forum)
  gb28181  - GB/T 28181
  rtmp     - RTMP (Real-Time Messaging Protocol)
  hls      - HLS (HTTP Live Streaming)

Examples:
  python server.py
  python server.py --source rtsp://admin:admin@192.168.42.1:554/live
  python server.py --source rtmp://server/live/stream
  python server.py --source http://server/live/stream.m3u8
  python server.py --port 9090
        """
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", default=8080, type=int,
                        help="Port to listen on (default: 8080)")
    parser.add_argument("--source", default=DEFAULT_URL,
                        help="Default video source URL")
    args = parser.parse_args()

    DEFAULT_URL = args.source

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/protocols", protocols)
    app.router.add_post("/offer", offer)

    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info(f"Default source: {DEFAULT_URL}")
    logger.info(f"Supported protocols: {', '.join(PROTOCOL_REGISTRY.keys())}")
    logger.info(f"MQTT Broker: {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}, topic: {MQTT_TOPIC}")
    logger.info(f"Open http://localhost:{args.port} in your browser")

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()