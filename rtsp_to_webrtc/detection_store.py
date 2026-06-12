"""
共享检测结果存储模块。

server.py 写入检测结果，video_sources.py 读取检测结果。
避免两者之间的循环导入。

支持检测结果历史记录，用于解决检测框与视频画面不同步的问题。
"""

import collections
import threading
import time

# 结构化的检测结果
# 格式: {"boxes": [...], "labels": [...], "resolution": [640, 640], "timestamp": float}
latest_detections = None
detections_lock = threading.Lock()

# 检测结果历史缓冲区（保留最近 2 秒的记录）
# 用于匹配延迟的视频帧
_detection_history = collections.deque()
_DETECTION_HISTORY_MAX_AGE = 2.0  # 秒


def set_detections(boxes, labels, resolution):
    """更新检测结果（线程安全），附带时间戳并写入历史记录。"""
    global latest_detections
    now = time.monotonic()
    detection = {
        "boxes": boxes,
        "labels": labels,
        "resolution": resolution,
        "timestamp": now,
    }
    with detections_lock:
        latest_detections = detection
        _detection_history.append(detection)
        # 清除超过最大年龄的旧记录
        cutoff = now - _DETECTION_HISTORY_MAX_AGE
        while len(_detection_history) > 1 and _detection_history[0]["timestamp"] < cutoff:
            _detection_history.popleft()


def get_detections():
    """获取最新的检测结果（线程安全）。"""
    with detections_lock:
        return latest_detections


def get_detections_at(target_timestamp):
    """
    获取最接近 target_timestamp 时刻的历史检测结果。

    用于延迟匹配：视频帧比检测结果延迟到达，
    通过取 target_timestamp = now - frame_delay_ms 的检测结果，
    使检测框与视频画面对齐。

    Args:
        target_timestamp: 目标时间戳（monotonic 时钟）

    Returns:
        最接近的检测结果 dict，若无匹配则返回 None。
    """
    with detections_lock:
        if not _detection_history:
            return latest_detections

        best = None
        best_dt = float("inf")

        # 从尾部向前搜索（deque 按时间递增，尾部最新）
        for det in reversed(_detection_history):
            dt = abs(det["timestamp"] - target_timestamp)
            if dt < best_dt:
                best_dt = dt
                best = det
            else:
                # deque 有序，越往前时间越早，差距只会越大
                break

        return best