"""
Embedded MQTT Broker and Subscriber Client.

Provides a lightweight MQTT 3.1.1 Broker (pure asyncio) and a paho-mqtt
subscriber client. Receives YOLO inference results from node-red and
stores them in detection_store for video overlay rendering.
"""

import asyncio
import json
import logging
import struct
import threading

import paho.mqtt.client as mqtt

from detection_store import set_detections

logger = logging.getLogger("mqtt-broker")

# ============================================================================
# Configuration
# ============================================================================

MQTT_BROKER_HOST = "0.0.0.0"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "yolo11n_result"

# Latest raw MQTT message (thread-safe)
latest_mqtt_message = None
mqtt_message_lock = threading.Lock()


# ============================================================================
# Lightweight MQTT 3.1.1 Broker (pure asyncio)
# ============================================================================

class SimpleMQTTBroker:
    """
    Lightweight MQTT 3.1.1 Broker, pure asyncio implementation.
    Supports basic CONNECT/PUBLISH/SUBSCRIBE/PINGREQ/DISCONNECT.
    Sufficient for node-red to publish inference results and internal
    client subscriptions.
    """

    def __init__(self, host="0.0.0.0", port=1883):
        self.host = host
        self.port = port
        self.server = None
        self.clients = {}  # writer -> {"subscriptions": set, "client_id": str}
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the MQTT Broker TCP server."""
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        addrs = ", ".join(str(s.getsockname()) for s in self.server.sockets)
        logger.info(f"[MQTT Broker] Started, listening on {addrs}")
        logger.info("[MQTT Broker] node-red can connect to publish inference results")
        return self

    async def shutdown(self):
        """Shut down the Broker."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            for writer in list(self.clients.keys()):
                try:
                    writer.close()
                except Exception:
                    pass
            self.clients.clear()
            logger.info("[MQTT Broker] Shut down")

    async def _handle_client(self, reader, writer):
        """Handle a single MQTT client connection."""
        addr = writer.get_extra_info("peername")
        client_info = {"subscriptions": set(), "client_id": f"unknown-{addr}"}
        self.clients[writer] = client_info
        logger.info(f"[MQTT Broker] Client connected: {addr}")

        try:
            while True:
                first_byte = await reader.read(1)
                if not first_byte:
                    break

                packet_type = (first_byte[0] >> 4) & 0x0F

                remaining_length = 0
                multiplier = 1
                for _ in range(4):
                    b = await reader.read(1)
                    if not b:
                        return
                    byte_val = b[0]
                    remaining_length += (byte_val & 0x7F) * multiplier
                    multiplier *= 128
                    if (byte_val & 0x80) == 0:
                        break

                data = b""
                while len(data) < remaining_length:
                    chunk = await reader.read(remaining_length - len(data))
                    if not chunk:
                        return
                    data += chunk

                if packet_type == 1:  # CONNECT
                    await self._handle_connect(writer, data, client_info)
                elif packet_type == 3:  # PUBLISH
                    await self._handle_publish(writer, first_byte[0], data)
                elif packet_type == 8:  # SUBSCRIBE
                    await self._handle_subscribe(writer, data, client_info)
                elif packet_type == 12:  # PINGREQ
                    await self._send_pingresp(writer)
                elif packet_type == 14:  # DISCONNECT
                    logger.info(f"[MQTT Broker] Client {client_info['client_id']} disconnected")
                    break
                else:
                    logger.debug(f"[MQTT Broker] Unhandled packet type: {packet_type}")

        except (ConnectionResetError, asyncio.IncompleteReadError, Exception) as e:
            logger.debug(f"[MQTT Broker] Client {addr} connection error: {e}")
        finally:
            del self.clients[writer]
            try:
                writer.close()
            except Exception:
                pass
            logger.info(f"[MQTT Broker] Client disconnected: {addr}")

    async def _handle_connect(self, writer, data, client_info):
        """Handle CONNECT packet, return CONNACK."""
        try:
            proto_len = struct.unpack("!H", data[0:2])[0]
            proto_name = data[2:2 + proto_len].decode("utf-8")
            proto_level = data[2 + proto_len]

            connect_flags = data[2 + proto_len + 1]

            payload_start = 2 + proto_len + 4
            if payload_start < len(data):
                client_id_len = struct.unpack("!H", data[payload_start:payload_start + 2])[0]
                client_id = data[payload_start + 2:payload_start + 2 + client_id_len].decode("utf-8")
                client_info["client_id"] = client_id or f"client-{id(writer)}"
            else:
                client_info["client_id"] = f"client-{id(writer)}"

            logger.info(f"[MQTT Broker] CONNECT: client_id={client_info['client_id']}, "
                        f"protocol={proto_name} v{proto_level}")

            connack = bytes([0x20, 0x02, 0x00, 0x00])
            writer.write(connack)
            await writer.drain()

        except Exception as e:
            logger.error(f"[MQTT Broker] CONNECT error: {e}")

    async def _handle_publish(self, writer, first_byte, data):
        """Handle PUBLISH packet, forward to all subscribers."""
        qos = (first_byte >> 1) & 0x03
        dup = (first_byte >> 3) & 0x01
        retain = first_byte & 0x01

        try:
            topic_len = struct.unpack("!H", data[0:2])[0]
            topic = data[2:2 + topic_len].decode("utf-8")

            offset = 2 + topic_len

            packet_id = None
            if qos > 0:
                packet_id = struct.unpack("!H", data[offset:offset + 2])[0]
                offset += 2

            payload = data[offset:]
            payload_str = payload.decode("utf-8", errors="replace")

            logger.debug(f"[MQTT Broker] PUBLISH: topic={topic}, qos={qos}, "
                         f"dup={dup}, retain={retain}, payload_len={len(payload)}")

            # Parse JSON and extract detection results
            parsed = None
            try:
                parsed = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.info(f"[MQTT] Non-JSON message: {payload_str}")

            if isinstance(parsed, dict):
                # Support reCamera YOLO11n format: {code, data: {boxes, labels, resolution}}
                det_data = parsed.get("data", parsed)
                boxes = det_data.get("boxes", [])
                labels = det_data.get("labels", [])
                resolution = det_data.get("resolution", [640, 640])

                set_detections(boxes, labels, resolution)

                global latest_mqtt_message
                with mqtt_message_lock:
                    latest_mqtt_message = payload_str

                if boxes:
                    logger.info(f"[MQTT] Detected {len(boxes)} target(s):")
                    for i, (box, label) in enumerate(zip(boxes, labels)):
                        logger.info(f"  Target {i+1}: {label} - bbox: {box}")

            # Forward to subscribers
            await self._forward_publish(topic, first_byte, data)

            # QoS 1: PUBACK
            if qos == 1 and packet_id is not None:
                puback = bytes([0x40, 0x02]) + struct.pack("!H", packet_id)
                writer.write(puback)
                await writer.drain()

            # QoS 2: PUBREC
            elif qos == 2 and packet_id is not None:
                pubrec = bytes([0x50, 0x02]) + struct.pack("!H", packet_id)
                writer.write(pubrec)
                await writer.drain()

        except Exception as e:
            logger.error(f"[MQTT Broker] PUBLISH error: {e}")

    async def _forward_publish(self, topic, first_byte, data):
        """Forward PUBLISH to clients subscribed to matching topics."""
        async with self._lock:
            for w, info in list(self.clients.items()):
                for sub_topic in info.get("subscriptions", set()):
                    if self._topic_matches(sub_topic, topic):
                        try:
                            w.write(bytes([first_byte]) + self._encode_remaining_length(len(data)) + data)
                            await w.drain()
                        except Exception:
                            pass

    async def _handle_subscribe(self, writer, data, client_info):
        """Handle SUBSCRIBE packet, return SUBACK."""
        try:
            packet_id = struct.unpack("!H", data[0:2])[0]
            offset = 2

            granted_qos = []
            while offset < len(data):
                topic_filter_len = struct.unpack("!H", data[offset:offset + 2])[0]
                topic_filter = data[offset + 2:offset + 2 + topic_filter_len].decode("utf-8")
                qos_requested = data[offset + 2 + topic_filter_len]
                offset += 2 + topic_filter_len + 1

                client_info["subscriptions"].add(topic_filter)
                granted_qos.append(qos_requested)
                logger.info(f"[MQTT Broker] SUBSCRIBE: client={client_info['client_id']}, "
                            f"topic={topic_filter}, qos={qos_requested}")

            suback_payload = struct.pack("!H", packet_id) + bytes(granted_qos)
            suback = bytes([0x90]) + self._encode_remaining_length(len(suback_payload)) + suback_payload
            writer.write(suback)
            await writer.drain()

        except Exception as e:
            logger.error(f"[MQTT Broker] SUBSCRIBE error: {e}")

    async def _send_pingresp(self, writer):
        """Send PINGRESP."""
        writer.write(bytes([0xD0, 0x00]))
        await writer.drain()

    @staticmethod
    def _encode_remaining_length(length):
        """Encode MQTT remaining length field."""
        result = bytearray()
        while True:
            byte = length % 128
            length = length // 128
            if length > 0:
                byte |= 0x80
            result.append(byte)
            if length == 0:
                break
        return bytes(result)

    @staticmethod
    def _topic_matches(subscription, topic):
        """Simple MQTT topic matching (supports # and + wildcards)."""
        if subscription == topic:
            return True
        if subscription == "#":
            return True
        sub_parts = subscription.split("/")
        topic_parts = topic.split("/")
        for i, sub_part in enumerate(sub_parts):
            if sub_part == "#":
                return True
            if i >= len(topic_parts):
                return False
            if sub_part != "+" and sub_part != topic_parts[i]:
                return False
        return len(sub_parts) == len(topic_parts)


# ============================================================================
# MQTT Broker Start Helper
# ============================================================================

async def start_mqtt_broker():
    """Start the embedded MQTT Broker server."""
    try:
        broker = SimpleMQTTBroker(host=MQTT_BROKER_HOST, port=MQTT_BROKER_PORT)
        await broker.start()
        return broker
    except Exception as e:
        logger.error(f"[MQTT Broker] Failed to start: {e}")
        return None


# ============================================================================
# MQTT Subscriber Client (paho-mqtt, subscribes to local Broker)
# ============================================================================

def _on_connect(client, userdata, flags, rc):
    """MQTT client connected callback."""
    if rc == 0:
        logger.info(f"[MQTT Client] Connected to local Broker (localhost:{MQTT_BROKER_PORT})")
        client.subscribe(MQTT_TOPIC)
        logger.info(f"[MQTT Client] Subscribed to topic: {MQTT_TOPIC}")
    else:
        logger.error(f"[MQTT Client] Connection failed, rc={rc}")


def _on_message(client, userdata, msg):
    """MQTT message received callback (backup logging)."""
    global latest_mqtt_message
    try:
        payload = msg.payload.decode("utf-8")
        with mqtt_message_lock:
            latest_mqtt_message = payload
    except Exception as e:
        logger.error(f"[MQTT Client] Message error: {e}")


def _on_disconnect(client, userdata, rc):
    """MQTT client disconnected callback."""
    if rc != 0:
        logger.warning(f"[MQTT Client] Unexpected disconnect (rc={rc}), will auto-reconnect")


def _on_subscribe(client, userdata, mid, granted_qos):
    """Subscription confirmed callback."""
    logger.info(f"[MQTT Client] Subscription confirmed (mid={mid}, qos={granted_qos})")


def start_mqtt_subscriber():
    """Create and start the MQTT subscriber client (background thread)."""
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.on_disconnect = _on_disconnect
    client.on_subscribe = _on_subscribe

    try:
        client.connect_async("localhost", MQTT_BROKER_PORT, keepalive=60)
        client.loop_start()
        logger.info("[MQTT Client] Subscriber started")
    except Exception as e:
        logger.error(f"[MQTT Client] Failed to start: {e}")

    return client