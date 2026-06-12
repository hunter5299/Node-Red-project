"""
Video Protocol to WebRTC Bridge - Video Source Abstraction Layer

Supported protocols:
  - RTSP (Real-Time Streaming Protocol)
  - ONVIF (Open Network Video Interface Forum)
  - GB/T 28181 (国标28181)
  - RTMP (Real-Time Messaging Protocol)
  - HLS (HTTP Live Streaming)

All video sources are abstracted as aiortc VideoStreamTrack objects,
so they can be directly added to a WebRTC peer connection.

Usage:
    from video_sources import create_video_source

    track = create_video_source("rtsp://admin:admin@192.168.42.1:554/live")
    pc.addTrack(track)
"""

import asyncio
import logging
import os
import subprocess
import threading
import time
import cv2
import numpy as np
from abc import ABC, abstractmethod
from aiortc import VideoStreamTrack
from av import VideoFrame
from detection_store import get_detections, get_detections_at

logger = logging.getLogger("video-sources")

# Suppress verbose FFmpeg/OpenCV warnings
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                       "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")


# ============================================================================
# Base Class
# ============================================================================

class BaseVideoSource(VideoStreamTrack, ABC):
    """
    Abstract base class for all video sources.
    Every video source must implement:
      - _connect(): Establish connection to the source
      - _read_frame() -> np.ndarray: Read a single BGR frame
      - _disconnect(): Release resources
      - _reconnect(): Handle reconnection
    """

    def __init__(self, url: str, **kwargs):
        super().__init__()
        self.url = url
        self.params = kwargs
        self.width = 640
        self.height = 480
        self.fps = 25
        self._connected = False
        self._stopped = False

        # =====================================================================
        # 检测框与视频画面同步 —— 延迟匹配检测结果
        # =====================================================================
        # 时序分析：
        #   摄像头 → 取帧 ─┬→ RTSP → 网络 → OpenCV → 本模块 recv()
        #                  │    (RTSP 传输延迟约 200~500ms)
        #                  │
        #                  └→ YOLO 推理 → MQTT → 本模块检测存储
        #                       (推理约 50~100ms)
        #
        #   检测结果到达时间 ≈ 摄像头取帧时刻 + 推理延迟 (~100ms)
        #   视频帧到达时间  ≈ 摄像头取帧时刻 + RTSP延迟 (~300ms)
        #
        # 因为 RTSP 延迟 > 推理延迟，当 recv() 取到一帧视频时，
        # 该帧对应的检测结果实际上已经在 frame_delay_ms 毫秒之前到达了。
        #
        # 解决方案：查找 (当前时间 - frame_delay_ms) 时刻的检测结果，
        # 而非最新检测结果，使检测框与视频画面精确对齐。
        #
        # 可通过环境变量 FRAME_DELAY_MS 调整延迟量，默认 200ms。
        # 增大值 → 框滞后于画面（框太旧）；减小值 → 框超前于画面（框太新）。
        # =====================================================================
        self.frame_delay_ms = int(os.environ.get("FRAME_DELAY_MS", "500"))

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._frame_event = threading.Event()

        self._connect()

        # Start background frame reader
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        logger.info(
            f"[{self.__class__.__name__}] Background frame reader started, "
            f"frame_delay={self.frame_delay_ms}ms"
        )

    def _reader_loop(self):
        """Background thread: continuously read frames, keep only the latest."""
        while not self._stopped:
            try:
                frame = self._read_frame()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                    self._frame_event.set()
                else:
                    time.sleep(0.001)  # Small sleep on failure to avoid busy loop
            except Exception as e:
                logger.error(f"Frame reader error: {e}")
                time.sleep(0.01)

    def _get_latest_frame(self) -> np.ndarray:
        """Get the latest available frame (non-blocking)."""
        with self._frame_lock:
            if self._latest_frame is not None:
                return self._latest_frame
        # Wait briefly for first frame
        self._frame_event.wait(timeout=2.0)
        with self._frame_lock:
            return self._latest_frame

    @abstractmethod
    def _connect(self):
        """Connect to the video source. Must set self.width, self.height, self.fps."""
        pass

    @abstractmethod
    def _read_frame(self) -> np.ndarray:
        """
        Read a single frame from the source.
        Returns a numpy array in BGR format (OpenCV convention).
        Returns None if reading fails.
        """
        pass

    @abstractmethod
    def _disconnect(self):
        """Release all resources."""
        pass

    def _reconnect(self):
        """Attempt to reconnect to the video source."""
        logger.warning(f"Reconnecting to {self.__class__.__name__}: {self.url}")
        self._disconnect()
        try:
            self._connect()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    # COCO 80 类别名称（与 YOLO11n 模型对应）
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """在画面上绘制 YOLO 检测结果（边界框 + 标签 + 置信度）。

        使用延迟匹配的检测结果：查找 frame_delay_ms 毫秒前的检测记录，
        使检测框与当前视频画面对齐（补偿 RTSP 传输延迟）。
        """
        # 取 (当前时间 - 延迟量) 时刻的检测结果，与当前视频帧对齐
        target_ts = time.monotonic() - self.frame_delay_ms / 1000.0
        detections = get_detections_at(target_ts)

        if not detections or not detections.get("boxes"):
            return frame

        boxes = detections["boxes"]
        resolution = detections.get("resolution", [640, 640])

        frame_h, frame_w = frame.shape[:2]
        model_w, model_h = resolution[0], resolution[1]

        # 计算缩放比例
        scale_x = frame_w / model_w
        scale_y = frame_h / model_h

        # 颜色表（BGR 格式）
        colors = [
            (0, 0, 255),    # 红
            (0, 255, 0),    # 绿
            (0, 255, 255),  # 黄
            (0, 165, 255),  # 橙
            (255, 0, 0),    # 蓝
            (255, 0, 255),  # 紫
            (255, 255, 0),  # 青
        ]

        for box in boxes:
            # 解析边界框: [cx, cy, w, h, score, label_idx]
            if not isinstance(box, (list, tuple)) or len(box) < 6:
                continue

            cx, cy, w, h = box[0], box[1], box[2], box[3]
            score = box[4]          # 置信度（可能是 0~1 或 0~100）
            label_idx = int(box[5]) # 类别索引

            # 将中心点+宽高转换为左上角+右下角坐标（模型分辨率空间）
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # 缩放坐标到实际画面尺寸
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # 确保坐标在画面范围内
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            # 获取类别名称
            if 0 <= label_idx < len(self.COCO_CLASSES):
                class_name = self.COCO_CLASSES[label_idx]
            else:
                class_name = f"class_{label_idx}"

            # 置信度文本：处理 0~1 或 0~100 格式
            if score > 1:
                confidence = score / 100.0
            else:
                confidence = score
            label_text = f"{class_name} {confidence:.0%}"

            color = colors[label_idx % len(colors)]

            # 画边界框 - 确保所有坐标为 Python int（避免 numpy 类型导致 OpenCV 报错）
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            thickness = max(2, int(min(frame_w, frame_h) / 300))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 准备绘制文字
            font_scale = max(0.5, min(frame_w, frame_h) / 1000)
            font_thickness = max(1, int(font_scale * 2))

            (text_w, text_h), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            text_w, text_h, baseline = int(text_w), int(text_h), int(baseline)

            # 标签背景框（放在边界框上方，避免超出顶部）
            label_y = int(max(y1 - text_h - baseline - 6, 0))
            cv2.rectangle(frame, (x1, label_y), (int(x1 + text_w + 6), y1), color, -1)

            # 标签文字（白色）
            cv2.putText(
                frame, label_text,
                (int(x1 + 3), int(y1 - baseline - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), font_thickness, cv2.LINE_AA,
            )

        return frame

    async def recv(self):
        """
        Called by aiortc to get the next video frame.
        Uses the latest frame from the background reader (zero-wait).
        This eliminates frame queue buildup and minimizes latency.

        检测框通过延迟匹配与视频画面同步：
        在 _draw_detections 中查找 frame_delay_ms 毫秒前的检测结果，
        补偿 RTSP 传输延迟，使框与画面精确对齐。
        """
        pts, time_base = await self.next_timestamp()

        # Get latest frame (non-blocking, always the freshest frame)
        frame = self._get_latest_frame()

        if frame is None:
            logger.warning("Failed to read frame, returning black frame")
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 使用帧的实际分辨率（避免不必要的 resize 导致模糊）
        h, w = frame.shape[:2]
        if (w, h) != (self.width, self.height):
            # 帧尺寸与预期不符时，更新 self.width/height 以匹配实际帧
            # 而不是缩放帧（缩放会导致画面模糊）
            self.width = w
            self.height = h

        # 确保帧是可写的（OpenCV cap.read() 可能返回只读数组）
        frame = frame.copy()

        # 绘制 YOLO 检测结果（延迟匹配，框与画面对齐）
        frame = self._draw_detections(frame)

        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create VideoFrame for WebRTC
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base

        return video_frame

    def stop(self):
        """Stop the track and release resources."""
        self._stopped = True
        super().stop()
        self._disconnect()
        self._connected = False
        logger.info(f"{self.__class__.__name__} stopped")


# ============================================================================
# RTSP Source
# ============================================================================

class RTSPSource(BaseVideoSource):
    """
    RTSP video source using OpenCV + FFmpeg.

    Args:
        url: RTSP stream URL (e.g., rtsp://admin:admin@192.168.42.1:554/live)
        transport: Transport protocol ('tcp' or 'udp'), default 'tcp'
    """

    def __init__(self, url: str, transport: str = "tcp", **kwargs):
        self.transport = transport
        self.cap = None
        super().__init__(url, **kwargs)

    def _connect(self):
        # Set transport via environment variable
        if self.transport == "tcp":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
            )
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;udp|fflags;nobuffer|flags;low_delay"
            )

        logger.info(f"[RTSP] Connecting to {self.url}")
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(f"[RTSP] Cannot connect to {self.url}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self._connected = True
        logger.info(f"[RTSP] Connected: {self.width}x{self.height} @ {self.fps}fps")

    def _read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            self._reconnect()
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def _disconnect(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# ONVIF Source
# ============================================================================

class ONVIFSource(BaseVideoSource):
    """
    ONVIF video source. Discovers the RTSP stream URL from an ONVIF device,
    then uses RTSP to receive the video.

    Args:
        url: ONVIF device URL (e.g., http://192.168.42.1:80/onvif/device_service)
        username: ONVIF username
        password: ONVIF password
        profile_token: ONVIF profile token (optional, uses first profile if not specified)

    Requires: onvif-zeep package (pip install onvif-zeep)
    """

    def __init__(self, url: str, username: str = "admin", password: str = "admin",
                 profile_token: str = None, **kwargs):
        self.username = username
        self.password = password
        self.profile_token = profile_token
        self.cap = None
        self._rtsp_url = None
        super().__init__(url, **kwargs)

    def _connect(self):
        logger.info(f"[ONVIF] Discovering RTSP URL from {self.url}")

        try:
            from onvif import ONVIFCamera

            # Parse host and port from URL
            url_clean = self.url.replace("http://", "").replace("https://", "")
            parts = url_clean.split("/")
            host_port = parts[0]
            if ":" in host_port:
                host, port = host_port.split(":")
                port = int(port)
            else:
                host = host_port
                port = 80

            cam = ONVIFCamera(host, port, self.username, self.password)

            # Get media service
            media_service = cam.create_media_service()

            # Get profiles
            profiles = media_service.GetProfiles()

            if not profiles:
                raise ConnectionError("[ONVIF] No profiles found on device")

            # Select profile
            if self.profile_token:
                profile = next(
                    (p for p in profiles if p.token == self.profile_token),
                    profiles[0]
                )
            else:
                profile = profiles[0]

            # Get stream URI
            stream_setup = {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            }
            uri_response = media_service.GetStreamUri({
                'StreamSetup': stream_setup,
                'ProfileToken': profile.token
            })
            self._rtsp_url = uri_response.Uri

            logger.info(f"[ONVIF] Discovered RTSP URL: {self._rtsp_url}")

        except ImportError:
            logger.warning("[ONVIF] onvif-zeep not installed, falling back to direct RTSP")
            # Fallback: construct RTSP URL from ONVIF URL
            url_clean = self.url.replace("http://", "").replace("https://", "")
            host_port = url_clean.split("/")[0]
            host = host_port.split(":")[0]
            self._rtsp_url = f"rtsp://{self.username}:{self.password}@{host}:554/live"
            logger.info(f"[ONVIF] Fallback RTSP URL: {self._rtsp_url}")
        except Exception as e:
            raise ConnectionError(f"[ONVIF] Discovery failed: {e}")

        # Use RTSP to actually receive the stream
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"
        )
        self.cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(f"[ONVIF] Cannot connect to RTSP stream: {self._rtsp_url}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self._connected = True
        logger.info(f"[ONVIF] Connected: {self.width}x{self.height} @ {self.fps}fps")

    def _read_frame(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret:
            self._reconnect()
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def _disconnect(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# GB/T 28181 Source (国标28181)
# ============================================================================

class GB28181Source(BaseVideoSource):
    """
    GB/T 28181 video source (国标28181).

    GB28181 is a Chinese national standard for video surveillance.
    It uses SIP signaling + RTP/PS streaming.

    Typical flow:
      1. Register with SIP server
      2. Send INVITE to request video stream
      3. Receive RTP/PS stream
      4. Convert PS to raw video via FFmpeg

    Args:
        url: Device ID or SIP URI (e.g., sip:34020000001320000001@192.168.1.1)
        server_ip: SIP server IP
        server_port: SIP server port (default 5060)
        device_id: GB28181 device ID (20 digits)
        username: SIP username
        password: SIP password
        local_ip: Local IP address

    Note: This implementation uses FFmpeg subprocess to receive the stream.
    For production, consider using specialized GB28181 libraries.
    """

    def __init__(self, url: str, server_ip: str = "", server_port: int = 5060,
                 device_id: str = "", username: str = "", password: str = "",
                 local_ip: str = "", **kwargs):
        self.server_ip = server_ip
        self.server_port = server_port
        self.device_id = device_id or url.split("@")[0].split(":")[-1]
        self.sip_username = username
        self.sip_password = password
        self.local_ip = local_ip
        self.cap = None
        self._ffmpeg_proc = None
        super().__init__(url, **kwargs)

    def _connect(self):
        logger.info(f"[GB28181] Connecting to device {self.device_id}")

        # GB28181 typically involves SIP signaling to invite the stream.
        # The actual RTP stream is received on a specific port.
        # Here we use FFmpeg to handle the SIP/RTP/PS conversion.
        #
        # In a real deployment, you would:
        # 1. Use a SIP stack to send INVITE to the device via the SIP server
        # 2. The device sends RTP/PS stream to your local IP:port
        # 3. FFmpeg demuxes PS and decodes video
        #
        # For now, this demonstrates the FFmpeg pipe approach.

        if not self.local_ip:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            self.local_ip = s.getsockname()[0]
            s.close()

        # FFmpeg command for GB28181 RTP/PS stream
        # This is a placeholder - actual implementation depends on your SIP server setup
        ffmpeg_cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", f"rtsp://{self.server_ip}:554/{self.device_id}",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an", "-sn",
            "-"
        ]

        logger.info(f"[GB28181] Starting FFmpeg: {' '.join(ffmpeg_cmd)}")

        try:
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            self.width = 1920
            self.height = 1080
            self.fps = 25
            self._connected = True
            logger.info(f"[GB28181] Connected (FFmpeg pipe mode)")
        except FileNotFoundError:
            logger.warning("[GB28181] FFmpeg not found, falling back to OpenCV")
            # Fallback to RTSP if FFmpeg is not available
            fallback_url = f"rtsp://{self.server_ip}:554/{self.device_id}"
            self.cap = cv2.VideoCapture(fallback_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.cap.isOpened():
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
                self._connected = True
                logger.info(f"[GB28181] Fallback RTSP: {self.width}x{self.height} @ {self.fps}fps")
            else:
                raise ConnectionError(f"[GB28181] Cannot connect to {fallback_url}")

    def _read_frame(self) -> np.ndarray:
        if self._ffmpeg_proc:
            # Read raw BGR frame from FFmpeg stdout
            frame_size = self.width * self.height * 3
            raw = self._ffmpeg_proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                return None
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            return frame
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        return None

    def _disconnect(self):
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# RTMP Source
# ============================================================================

class RTMPSource(BaseVideoSource):
    """
    RTMP video source using FFmpeg subprocess.

    Args:
        url: RTMP stream URL (e.g., rtmp://server/live/stream)
        use_ffmpeg: Use FFmpeg subprocess (True) or OpenCV (False), default True
    """

    def __init__(self, url: str, use_ffmpeg: bool = True, **kwargs):
        self.use_ffmpeg = use_ffmpeg
        self._ffmpeg_proc = None
        self.cap = None
        super().__init__(url, **kwargs)

    def _connect(self):
        logger.info(f"[RTMP] Connecting to {self.url}")

        if self.use_ffmpeg:
            self._connect_ffmpeg()
        else:
            self._connect_opencv()

        self._connected = True
        logger.info(f"[RTMP] Connected: {self.width}x{self.height} @ {self.fps}fps")

    def _connect_ffmpeg(self):
        """Connect using FFmpeg subprocess pipe."""
        ffmpeg_cmd = [
            "ffmpeg",
            "-rtmp_live", "live",
            "-i", self.url,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an", "-sn",
            "-"
        ]

        logger.info(f"[RTMP] Starting FFmpeg: {' '.join(ffmpeg_cmd)}")

        try:
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            # Default resolution, will be adjusted on first frame
            self.width = 1920
            self.height = 1080
            self.fps = 25
        except FileNotFoundError:
            logger.warning("[RTMP] FFmpeg not found, falling back to OpenCV")
            self.use_ffmpeg = False
            self._connect_opencv()

    def _connect_opencv(self):
        """Connect using OpenCV (FFmpeg backend)."""
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(f"[RTMP] Cannot connect to {self.url}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

    def _read_frame(self) -> np.ndarray:
        if self.use_ffmpeg and self._ffmpeg_proc:
            frame_size = self.width * self.height * 3
            raw = self._ffmpeg_proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                self._reconnect()
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self._reconnect()
                return None
            return frame
        return None

    def _disconnect(self):
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# HLS Source
# ============================================================================

class HLSSource(BaseVideoSource):
    """
    HLS (HTTP Live Streaming) video source using FFmpeg or OpenCV.

    Args:
        url: HLS stream URL (e.g., http://server/live/stream.m3u8)
        use_ffmpeg: Use FFmpeg subprocess (True) or OpenCV (False), default True
        live_latency: Target latency in seconds for live streams (default 3)
    """

    def __init__(self, url: str, use_ffmpeg: bool = True, live_latency: float = 3.0, **kwargs):
        self.use_ffmpeg = use_ffmpeg
        self.live_latency = live_latency
        self._ffmpeg_proc = None
        self.cap = None
        super().__init__(url, **kwargs)

    def _connect(self):
        logger.info(f"[HLS] Connecting to {self.url}")

        if self.use_ffmpeg:
            self._connect_ffmpeg()
        else:
            self._connect_opencv()

        self._connected = True
        logger.info(f"[HLS] Connected: {self.width}x{self.height} @ {self.fps}fps")

    def _connect_ffmpeg(self):
        """Connect using FFmpeg subprocess pipe."""
        ffmpeg_cmd = [
            "ffmpeg",
            "-live_start_index", str(-int(self.live_latency)),
            "-i", self.url,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an", "-sn",
            "-"
        ]

        logger.info(f"[HLS] Starting FFmpeg: {' '.join(ffmpeg_cmd)}")

        try:
            self._ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            self.width = 1920
            self.height = 1080
            self.fps = 25
        except FileNotFoundError:
            logger.warning("[HLS] FFmpeg not found, falling back to OpenCV")
            self.use_ffmpeg = False
            self._connect_opencv()

    def _connect_opencv(self):
        """Connect using OpenCV (FFmpeg backend)."""
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise ConnectionError(f"[HLS] Cannot connect to {self.url}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25

    def _read_frame(self) -> np.ndarray:
        if self.use_ffmpeg and self._ffmpeg_proc:
            frame_size = self.width * self.height * 3
            raw = self._ffmpeg_proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                self._reconnect()
                return None
            return np.frombuffer(raw, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
        elif self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self._reconnect()
                return None
            return frame
        return None

    def _disconnect(self):
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            self._ffmpeg_proc = None
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# Factory Function
# ============================================================================

# Protocol registry
PROTOCOL_REGISTRY = {
    "rtsp": RTSPSource,
    "onvif": ONVIFSource,
    "gb28181": GB28181Source,
    "rtmp": RTMPSource,
    "hls": HLSSource,
}

# URL prefix to protocol mapping
URL_PROTOCOL_MAP = {
    "rtsp://": "rtsp",
    "rtmp://": "rtmp",
    "http://": "hls",      # Default HTTP to HLS
    "https://": "hls",     # Default HTTPS to HLS
    "onvif://": "onvif",
    "onvif:": "onvif",
    "gb28181://": "gb28181",
    "gb28181:": "gb28181",
    "sip:": "gb28181",
}


def detect_protocol(url: str) -> str:
    """Auto-detect video protocol from URL."""
    url_lower = url.lower()
    for prefix, protocol in URL_PROTOCOL_MAP.items():
        if url_lower.startswith(prefix):
            # Special case: HTTP(S) URLs ending with .m3u8 are definitely HLS
            if protocol == "hls" and not url_lower.endswith(".m3u8"):
                # Could also be ONVIF HTTP endpoint
                if "onvif" in url_lower:
                    return "onvif"
            return protocol
    raise ValueError(
        f"Unknown protocol for URL: {url}\n"
        f"Supported: {', '.join(PROTOCOL_REGISTRY.keys())}"
    )


def create_video_source(url: str, protocol: str = None, **kwargs) -> BaseVideoSource:
    """
    Create a video source track from a URL.

    Args:
        url: Video stream URL
        protocol: Force a specific protocol ('rtsp', 'onvif', 'gb28181', 'rtmp', 'hls')
                  If None, auto-detected from URL.
        **kwargs: Additional parameters passed to the source constructor.

    Returns:
        A VideoStreamTrack that can be added to a WebRTC peer connection.

    Examples:
        # RTSP (auto-detected)
        track = create_video_source("rtsp://admin:admin@192.168.42.1:554/live")

        # RTMP (auto-detected)
        track = create_video_source("rtmp://server/live/stream")

        # HLS (auto-detected)
        track = create_video_source("http://server/live/stream.m3u8")

        # ONVIF (explicit protocol)
        track = create_video_source("http://192.168.1.100/onvif/device_service",
                                     protocol="onvif", username="admin", password="admin")

        # GB28181 (explicit protocol)
        track = create_video_source("sip:34020000001320000001@192.168.1.1",
                                     protocol="gb28181", server_ip="192.168.1.1")
    """
    if protocol is None:
        protocol = detect_protocol(url)

    protocol = protocol.lower()

    if protocol not in PROTOCOL_REGISTRY:
        raise ValueError(
            f"Unsupported protocol: {protocol}\n"
            f"Supported: {', '.join(PROTOCOL_REGISTRY.keys())}"
        )

    source_class = PROTOCOL_REGISTRY[protocol]
    logger.info(f"Creating {source_class.__name__} for {url}")

    return source_class(url=url, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def list_protocols():
    """Print all supported protocols and their descriptions."""
    print("\nSupported Video Protocols:\n")
    print(f"{'Protocol':<12} {'Class':<20} {'Description'}")
    print("-" * 70)
    for name, cls in PROTOCOL_REGISTRY.items():
        print(f"{name:<12} {cls.__name__:<20} {cls.__doc__.split(chr(10))[0].strip()}")
    print()


def get_stream_info(url: str, protocol: str = None) -> dict:
    """
    Get stream information without creating a full WebRTC track.
    Returns a dict with width, height, fps, protocol.
    """
    try:
        source = create_video_source(url, protocol=protocol)
        info = {
            "width": source.width,
            "height": source.height,
            "fps": source.fps,
            "protocol": source.__class__.__name__,
            "url": url,
        }
        source.stop()
        return info
    except Exception as e:
        return {"error": str(e), "url": url}


if __name__ == "__main__":
    # Demo: list supported protocols
    list_protocols()

    # Demo: detect protocol from URL
    test_urls = [
        "rtsp://admin:admin@192.168.42.1:554/live",
        "rtmp://server/live/stream",
        "http://server/live/stream.m3u8",
        "http://192.168.1.100/onvif/device_service",
        "sip:34020000001320000001@192.168.1.1",
    ]
    print("Protocol Detection Demo:\n")
    for url in test_urls:
        try:
            proto = detect_protocol(url)
            cls = PROTOCOL_REGISTRY[proto].__name__
            print(f"  {url}")
            print(f"    -> Protocol: {proto}, Class: {cls}\n")
        except ValueError as e:
            print(f"  {url}")
            print(f"    -> Error: {e}\n")