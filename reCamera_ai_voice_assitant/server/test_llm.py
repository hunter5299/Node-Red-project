"""
星火大模型 LLM 一次性测试脚本
测试前请确保 server.py 中的 LLM_APPID、LLM_APISecret、LLM_APIKey 已填入真实密钥
"""
import sys
import ssl
import json
import base64
import hashlib
import hmac
import _thread as thread
import websocket
from datetime import datetime
from urllib.parse import urlencode, urlparse

# 使用 certifi 提供的 CA 证书，解决 Windows 下 SSL 验证失败问题
import certifi

LLM_APPID = "da94379a"
LLM_APISecret = "NGRlZDk2MDY0ODA1NTViNGVhNTY2YmFm"
LLM_APIKey = "e23e27197669ca24bcc1f76574007dbe"


def create_url(api_url, api_secret, api_key):
    """生成科大讯飞 Websocket 请求的鉴权 URL"""
    url = urlparse(api_url)
    date = datetime.now(datetime.now().astimezone().tzinfo.__class__.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    signature_origin = f"host: {url.netloc}\ndate: {date}\nGET {url.path} HTTP/1.1"
    signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha256).digest()
    signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    v = {"authorization": authorization, "date": date, "host": url.netloc}
    return api_url + '?' + urlencode(v)


def run_llm(user_text):
    """调用星火大模型，返回生成的回答"""
    ws_url = create_url("wss://spark-api.xf-yun.com/v1.1/chat", LLM_APISecret, LLM_APIKey)
    result_text = []
    error_info = []

    def on_message(ws, message):
        msg = json.loads(message)
        code = msg['header']['code']
        if code != 0:
            error_info.append(f"错误码: {code}, 错误信息: {msg['header']['message']}")
            ws.close()
            return
        choices = msg['payload']['choices']
        content = choices['text'][0]['content']
        result_text.append(content)
        if choices['status'] == 2:
            ws.close()

    def on_error(ws, error):
        error_info.append(f"WebSocket 错误: {error}")

    def on_open(ws):
        def run(*args):
            data = {
                "header": {
                    "app_id": LLM_APPID,
                    "uid": "test_user"
                },
                "parameter": {
                    "chat": {
                        "domain": "lite",
                        "temperature": 0.5,
                        "max_tokens": 1024
                    }
                },
                "payload": {
                    "message": {
                        "text": [
                            {"role": "system",
                             "content": "你是一个智能助手，请用简短友好的语言回答问题,尽量50个子以内。"},
                            {"role": "user", "content": user_text}
                        ]
                    }
                }
            }
            ws.send(json.dumps(data))

        thread.start_new_thread(run, ())

    ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
    ws.on_open = on_open
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_REQUIRED, "ssl_version": ssl.PROTOCOL_TLS_CLIENT, "ca_certs": certifi.where()})

    if error_info:
        return None, error_info
    return "".join(result_text), []


if __name__ == '__main__':
    # 检查密钥是否已配置
    if LLM_APPID == "xxx" or LLM_APISecret == "xxx" or LLM_APIKey == "xxx":
        print("❌ 请先在脚本中配置真实的 LLM_APPID、LLM_APISecret、LLM_APIKey")
        print("   获取地址: https://www.xfyun.cn/services/SparkLLM")
        sys.exit(1)

    # 测试消息
    test_message = "你好，请介绍一下你自己。"

    print("=" * 50)
    print(f"📤 发送消息: {test_message}")
    print("=" * 50)

    result, errors = run_llm(test_message)

    if errors:
        print("\n❌ 调用失败:")
        for e in errors:
            print(f"   {e}")
    else:
        print(f"\n📥 收到回复: {result}")

    print("=" * 50)
