# 视频协议 WebRTC 播放器

将多种视频协议（RTSP / ONVIF / GB28181 / RTMP / HLS）转换为 WebRTC 格式，在浏览器中低延迟实时播放。

## 架构

```
视频源 (RTSP/ONVIF/GB28181/RTMP/HLS)
    → video_sources.py (协议抽象层，后台线程读取最新帧)
    → aiortc WebRTC 编码
    → 浏览器实时播放
```

## 延迟优化

采用**后台线程持续读取 + 只保留最新帧**策略：
- 后台线程持续从视频源读取帧，丢弃旧帧
- WebRTC `recv()` 始终获取最新帧，零等待
- 典型端到端延迟：**200-500ms**

## 支持的协议

| 协议 | URL 格式 | 说明 |
|------|----------|------|
| RTSP | `rtsp://...` | 实时流协议 (默认) |
| ONVIF | `http://...onvif...` | 自动发现 RTSP URL |
| GB/T 28181 | `sip:...` | 国标视频监控 |
| RTMP | `rtmp://...` | 实时消息协议 |
| HLS | `http://....m3u8` | HTTP 直播流 |

## 安装

```bash
cd rtsp_websiteshow
pip install -r requirements.txt
```

> **注意**: Windows 上安装 `aiortc` 可能需要先安装 Microsoft C++ Build Tools。
> ONVIF 支持可选安装: `pip install onvif-zeep`

## 使用方法

### 1. 启动服务器

```bash
python server.py
```

自定义参数：
```bash
python server.py --source rtsp://admin:admin@192.168.42.1:554/live --port 8080
python server.py --source rtmp://server/live/stream
python server.py --source http://server/live/stream.m3u8
```

### 2. 打开浏览器

访问 http://localhost:8080

### 3. 选择协议 + 播放

- 下拉选择协议（或自动检测）
- 输入视频流地址
- 点击"开始播放"

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 服务器绑定地址 |
| `--port` | `8080` | 服务器端口 |
| `--source` | `rtsp://admin:admin@192.168.42.1:554/live` | 默认视频源地址 |

## 文件说明

- `server.py` - WebRTC 信令服务器
- `video_sources.py` - 视频协议抽象层（5种协议 + 工厂函数）
- `index.html` - 前端播放器（协议选择 + WebRTC）
- `requirements.txt` - Python 依赖列表

## 故障排除

1. **RTSP 连接失败**: 确保摄像头 IP 可达，检查是否被其他程序占用（大多数摄像头仅支持1-2个并发连接）
2. **aiortc 安装失败**: 需要安装 C++ 编译工具 (Visual Studio Build Tools)
3. **non-existing PPS 错误**: 正常现象，等待关键帧（约1秒）后自动恢复
4. **端口被占用**: `python server.py --port 8081` 使用其他端口
5. **视频延迟高**: 已优化为后台线程读取最新帧，确保网络带宽充足