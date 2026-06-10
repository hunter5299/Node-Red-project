"""
科大讯飞 STT 语音识别测试脚本
用法: python test_stt.py [音频文件路径]
如果不传参数，会自动生成一段测试音频
"""
import sys
import ssl
import json
import base64
import hashlib
import hmac
import struct
import time
import _thread as thread
import websocket
from datetime import datetime, timezone
from urllib.parse import urlencode, urlparse

# 使用 certifi 提供的 CA 证书
import certifi

# ================= 配置区 =================
STT_APPID = "da94379a"
STT_APISecret = "NGRlZDk2MDY0ODA1NTViNGVhNTY2YmFm"
STT_APIKey = "e23e27197669ca24bcc1f76574007dbe"


def create_url(api_url, api_secret, api_key):
    """生成科大讯飞 Websocket 请求的鉴权 URL"""
    url = urlparse(api_url)
    date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    signature_origin = f"host: {url.netloc}\ndate: {date}\nGET {url.path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    v = {"authorization": authorization, "date": date, "host": url.netloc}
    return api_url + '?' + urlencode(v)


def generate_test_wav(filename="test_audio.wav", duration_sec=3, sample_rate=16000):
    """生成一段测试用的 WAV 音频文件（正弦波 440Hz，模拟声音）"""
    import math
    num_samples = sample_rate * duration_sec
    samples = bytearray()
    for i in range(num_samples):
        # 生成 440Hz 正弦波，16bit PCM
        value = int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.extend(struct.pack('<h', value))

    # WAV 文件头
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                             b'RIFF', 36 + len(samples), b'WAVE', b'fmt ', 16, 1, 1, sample_rate,
                             sample_rate * 2, 2, 16, b'data', len(samples))

    with open(filename, 'wb') as f:
        f.write(wav_header + samples)
    print(f"📝 已生成测试音频: {filename} ({len(wav_header) + len(samples)} 字节, {duration_sec}秒)")
    return filename


def run_stt(audio_data):
    """执行语音识别，返回文字"""
    ws_url = create_url("wss://iat-api.xfyun.cn/v2/iat", STT_APISecret, STT_APIKey)
    result_text = []
    error_info = []

    def on_message(ws, message):
        msg = json.loads(message)
        code = msg.get('code', 0)
        if code != 0:
            error_info.append(f"错误码: {code}, 错误信息: {msg.get('message', 'unknown')}")
            ws.close()
            return
        ws_data = msg['data']['result']['ws']
        for i in ws_data:
            for w in i['cw']:
                result_text.append(w['w'])
        if msg['data']['status'] == 2:
            ws.close()

    def on_error(ws, error):
        error_info.append(f"WebSocket 错误: {error}")

    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)

    def on_open(ws):
        def run(*args):
            status = 0
            chunk_size = 8000
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if i + chunk_size >= len(audio_data):
                    status = 2
                elif i > 0:
                    status = 1

                d = {"common": {"app_id": STT_APPID},
                     "business": {"domain": "iat", "language": "zh_cn", "accent": "mandarin",
                                  "vinfo": 1, "vad_eos": 10000},
                     "data": {"status": status, "format": "audio/L16;rate=16000",
                              "audio": base64.b64encode(chunk).decode('utf-8'), "encoding": "raw"}}
                ws.send(json.dumps(d))
                time.sleep(0.04)

        thread.start_new_thread(run, ())

    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ssl_version": ssl.PROTOCOL_TLS_CLIENT, "ca_certs": certifi.where()})

    if error_info:
        return None, error_info
    return "".join(result_text), []


if __name__ == '__main__':
    if STT_APPID == "xxx" or STT_APISecret == "xxx" or STT_APIKey == "xxx":
        print("❌ 请先在脚本中配置真实的 STT_APPID、STT_APISecret、STT_APIKey")
        sys.exit(1)

    # 获取音频文件路径
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # 没有提供文件，自动生成测试音频
        audio_file = generate_test_wav()

    print("=" * 50)
    print(f"📂 读取音频文件: {audio_file}")

    # 读取音频
    try:
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
    except FileNotFoundError:
        print(f"❌ 文件不存在: {audio_file}")
        sys.exit(1)

    # 去除 WAV 头获取裸 PCM
    pcm_data = audio_bytes[44:] if audio_bytes.startswith(b'RIFF') else audio_bytes
    print(f"📊 音频大小: {len(audio_bytes)} 字节 (PCM: {len(pcm_data)} 字节)")
    print(f"📤 开始语音识别...")
    print("=" * 50)

    result, errors = run_stt(pcm_data)

    if errors:
        print("\n❌ 识别失败:")
        for e in errors:
            print(f"   {e}")
    elif result:
        print(f"\n📥 识别结果: {result}")
    else:
        print("\n⚠️ 未识别到有效语音内容（音频可能是静音或噪音）")

    print("=" * 50)
