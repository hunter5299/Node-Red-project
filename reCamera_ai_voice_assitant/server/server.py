import base64
import hashlib
import hmac
import json
import ssl
import time
import _thread as thread
from datetime import datetime, timezone
from urllib.parse import urlencode
import websocket
from flask import Flask, request, send_file
import struct
import os

# Use certifi CA certificates to fix SSL verification failures on Windows
import certifi

app = Flask(__name__)

## Flow: STT -> LLM -> TTS
# ================= Configuration =================
# 1. STT (Speech-to-Text) Configuration
STT_APPID = "da94379a"
STT_APISecret = "NGRlZDk2MDY0ODA1NTViNGVhNTY2YmFm"
STT_APIKey = "e23e27197669ca24bcc1f76574007dbe"

# 2. TTS (Text-to-Speech) Configuration
TTS_APPID = "da94379a"
TTS_APISecret = "NGRlZDk2MDY0ODA1NTViNGVhNTY2YmFm"
TTS_APIKey = "e23e27197669ca24bcc1f76574007dbe"

# 3. LLM (Spark Lite) Configuration
LLM_APPID = "da94379a"
LLM_APISecret = "NGRlZDk2MDY0ODA1NTViNGVhNTY2YmFm"
LLM_APIKey = "e23e27197669ca24bcc1f76574007dbe"


# ================= Utility Functions =================
def create_url(api_url, api_secret, api_key):
    """Generate authenticated URL for iFlytek WebSocket requests"""
    from urllib.parse import urlparse
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


def pcm2wav(pcm_data, channels=1, sample_rate=16000, sample_width=2):
    """Convert raw PCM stream to WAV format for reCamera playback"""
    wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
                             b'RIFF', 36 + len(pcm_data), b'WAVE', b'fmt ', 16, 1, channels, sample_rate,
                             sample_rate * channels * sample_width, channels * sample_width, sample_width * 8, b'data',
                             len(pcm_data))
    return wav_header + pcm_data


# ================= AI Capability Core Logic =================
def run_stt(audio_data):
    """Execute speech recognition (ears), return recognized text"""
    ws_url = create_url("wss://iat-api.xfyun.cn/v2/iat", STT_APISecret, STT_APIKey)
    result_text = []

    def on_message(ws, message):
        msg = json.loads(message)
        if msg['code'] != 0:
            print(f"STT Error: {msg['code']} - {msg['message']}")
            ws.close()
            return
        ws_data = msg['data']['result']['ws']
        for i in ws_data:
            for w in i['cw']:
                result_text.append(w['w'])
        if msg['data']['status'] == 2:
            ws.close()

    ws = websocket.WebSocketApp(ws_url, on_message=on_message)

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
                     "business": {"domain": "iat", "language": "zh_cn", "accent": "mandarin", "vinfo": 1,
                                  "vad_eos": 10000},
                     "data": {"status": status, "format": "audio/L16;rate=16000",
                              "audio": base64.b64encode(chunk).decode('utf-8'), "encoding": "raw"}}
                ws.send(json.dumps(d))
                time.sleep(0.04)

        thread.start_new_thread(run, ())

    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ssl_version": ssl.PROTOCOL_TLS_CLIENT, "ca_certs": certifi.where()})
    return "".join(result_text)


def run_llm(user_text):
    """Execute LLM inference (brain), return generated response"""
    ws_url = create_url("wss://spark-api.xf-yun.com/v1.1/chat", LLM_APISecret, LLM_APIKey)
    result_text = []

    def on_message(ws, message):
        msg = json.loads(message)
        if msg['header']['code'] != 0:
            print(f"LLM Error: {msg['header']['code']} - {msg['header']['message']}")
            ws.close()
            return

        # Extract text fragments from model response
        choices = msg['payload']['choices']
        content = choices['text'][0]['content']
        result_text.append(content)

        # If status is 2, the LLM has finished responding
        if choices['status'] == 2:
            ws.close()

    ws = websocket.WebSocketApp(ws_url, on_message=on_message)

    def on_open(ws):
        # Construct request parameters for Spark LLM
        data = {
            "header": {
                "app_id": LLM_APPID,
                "uid": "recamera_user"
            },
            "parameter": {
                "chat": {
                    "domain": "lite",  # Core fix: Spark Lite's dedicated domain is "lite"
                    "temperature": 0.5,
                    "max_tokens": 1024
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "system",
                         "content": "You are a voice assistant deployed on the smart camera reCamera. Please answer user questions in short, conversational, and friendly language, keeping each response within 50 characters."},
                        {"role": "user", "content": user_text}
                    ]
                }
            }
        }
        ws.send(json.dumps(data))

    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ssl_version": ssl.PROTOCOL_TLS_CLIENT, "ca_certs": certifi.where()})
    return "".join(result_text)


def run_tts(text):
    """Execute text-to-speech (mouth), return WAV byte stream"""
    ws_url = create_url("wss://cbm01.cn-huabei-1.xf-yun.com/v1/private/mcd9m97e6", TTS_APISecret, TTS_APIKey)
    audio_result = bytearray()

    def on_message(ws, message):
        msg = json.loads(message)
        if msg['header']['code'] != 0:
            print(f"TTS Error: {msg['header']['code']} - {msg['header']['message']}")
            ws.close()
            return
        audio = msg['payload']['audio']['audio']
        audio_result.extend(base64.b64decode(audio))
        if msg['header']['status'] == 2:
            ws.close()

    ws = websocket.WebSocketApp(ws_url, on_message=on_message)

    def on_open(ws):
        def run(*args):
            d = {
                "header": {
                    "app_id": TTS_APPID,
                    "status": 2
                },
                "parameter": {
                    "tts": {
                        "vcn": "x5_EnUs_Lila_flow",	
                        "speed": 50,
                        "volume": 50,
                        "pitch": 50,
                        "audio": {
                            "encoding": "raw",
                            "sample_rate": 16000
                        }
                    }
                },
                "payload": {
                    "text": {
                        "encoding": "utf8",
                        "status": 2,
                        "text": base64.b64encode(text.encode('utf-8')).decode('utf-8')
                    }
                }
            }
            ws.send(json.dumps(d))

        thread.start_new_thread(run, ())

    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ssl_version": ssl.PROTOCOL_TLS_CLIENT, "ca_certs": certifi.where()})

    return pcm2wav(bytes(audio_result))


# ================= Flask API Endpoints =================
@app.route('/interact', methods=['POST'])
def interact():
    audio_bytes = request.data
    print(f"\n=====================================")
    print(f"✅ Received audio from reCamera, size: {len(audio_bytes)} bytes")

    # Strip WAV header to get raw PCM
    pcm_data = audio_bytes[44:] if audio_bytes.startswith(b'RIFF') else audio_bytes

    # 1. Speech to text
    user_text = run_stt(pcm_data)
    print(f"🗣️ Heard user say: {user_text}")

    if not user_text or len(user_text.strip()) == 0:
        reply_text = "Sorry, I didn't hear clearly. Could you say that again?"
        print(f"🤖 No clear speech detected, default response: {reply_text}")
    else:
        # 2. Call LLM to think
        print(f"🧠 LLM is thinking...")
        reply_text = run_llm(user_text)

        # Fallback: if LLM errors or returns empty, provide a voice prompt
        if not reply_text:
            reply_text = "Oops, my brain is temporarily offline. Please try again later."

        print(f"💡 LLM response: {reply_text}")

    # 3. Text to speech
    print(f"👄 Generating speech...")
    tts_wav_bytes = run_tts(reply_text)

    # 4. Return audio to hardware
    reply_path = "temp_reply.wav"
    with open(reply_path, "wb") as f:
        f.write(tts_wav_bytes)

    print(f"🚀 Speech delivery complete! Waiting for next interaction.")
    return send_file(reply_path, mimetype="audio/wav")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

