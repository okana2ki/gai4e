#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# voice_agent.py
# 音声対話エージェント: Voskで発話のエンドポイント検知を行い、
# Gemini API（チャット機能）に転送し、gTTS+playsoundで合成音声を再生しつつ、
# LED制御を行います。LEDデバイスが見つからない場合は警告し、対話は継続します。

import os
import sys
import json
import tempfile
import asyncio
import ctypes
import time

# ── 設定: スキャンタイムアウト（秒） ──
DEVICE_SCAN_TIMEOUT = 15

# ── Windows: COMをMTAモードで初期化し、ProactorEventLoopを設定（WinRT対策） ──
if os.name == 'nt':
    try:
        ctypes.windll.ole32.CoInitializeEx(None, 0x0)
    except Exception:
        pass
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

# ── イベントループを1回生成 ──
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from gtts import gTTS
try:
    from playsound3 import playsound
except ImportError:
    from playsound import playsound

from bleak import BleakClient, BleakScanner
from struct import pack

# ── LED制御用UUID ──
CORE_INDICATE_UUID = '72c90005-57a9-4d40-b746-534e22ec9f9e'
CORE_NOTIFY_UUID   = '72c90003-57a9-4d40-b746-534e22ec9f9e'
CORE_WRITE_UUID    = '72c90004-57a9-4d40-b746-534e22ec9f9e'

# グローバルLEDクライアント
LED_CLIENT = None

async def scan_led(prefix='MESH-100LE'):
    """近くのLEDを永続スキャン。スキャン結果をログ出力します"""
    while True:
        devices = await BleakScanner.discover()
        if devices:
            print("スキャン結果:")
            for d in devices:
                name = d.name or "<Unnamed>"
                print(f"  {name} [{d.address}]")
        for d in devices:
            if d.name and prefix in d.name:
                return d
        print(f"LEDデバイス({prefix})が見つかりません。再スキャンします...")
        await asyncio.sleep(2)

async def connect_led():
    """LEDモジュールをスキャンして接続、初期化コマンド送信後クライアントを返す"""
    print(f"LEDデバイスをスキャン中... (最大{DEVICE_SCAN_TIMEOUT}秒まで待機)")
    try:
        device = await asyncio.wait_for(scan_led(), timeout=DEVICE_SCAN_TIMEOUT)
    except asyncio.TimeoutError:
        print(f"LEDデバイスが{DEVICE_SCAN_TIMEOUT}秒以内に見つかりません。LED操作は無効になります。")
        return None
    print(f"Found LED: {device.name} [{device.address}]")
    client = BleakClient(device)
    await client.connect()
    await client.start_notify(CORE_NOTIFY_UUID, lambda s, d: None)
    await client.start_notify(CORE_INDICATE_UUID, lambda s, d: None)
    # 初期化コマンド: 制御モードへ
    init_payload = pack('<BBBB', 0, 2, 1, 3)
    try:
        await client.write_gatt_char(CORE_WRITE_UUID, init_payload, response=True)
        print("→ LED 初期化コマンド送信完了")
    except Exception as e:
        print(f"LED 初期化エラー: {e}")
    return client

# ── Vosk/SampleRate 設定 ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-ja-0.22")

import sounddevice as sd

# デフォルト入力デバイスのサンプルレートを自動取得し、失敗時は16000Hzをフォールバック
try:
    default_input_device = sd.default.device[0]
    device_info = sd.query_devices(default_input_device, 'input')
    SAMPLE_RATE = int(device_info['default_samplerate'])
    print(f"使用マイク ({device_info['name']}) のデフォルトサンプルレート: {SAMPLE_RATE} Hz")
except Exception:
    print("入力デバイス情報の取得に失敗。16000Hz を使用します。")
    SAMPLE_RATE = 16000

# ── 初期化: 音声認識 & ChatGPT ──
model      = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
api_client = genai.Client()
chat       = api_client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=(
            "あなたは気ままな猫です。名前はキティです。ユーザと楽しく会話します。"
            "ユーザの発話に対して必ずJSONで返してください："
            "{\"speech\":\"(テキスト)\",\"command\":\"(LED_ON,LED_OFF,GET_TEMPERATUREなど)または空文字列\"}"
            "暗いときや明かりをつけたいときはLED_ON、消灯はLED_OFFを返してください。"
        ),
        max_output_tokens=80
    )
)

def speak_gtts(text, lang="ja"):
    """
    gTTSでテキストを音声合成し、一時ファイルを再生後に削除
    """
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        path = f.name
    playsound(path)
    os.remove(path)


def parse_and_dispatch(raw_text):
    """
    raw_textはJSON{"speech":"...","command":"..."}を想定
    """
    try:
        obj = json.loads(raw_text)
        speech = obj.get("speech", "")
        cmd    = obj.get("command", "")
    except:
        speech, cmd = raw_text, ""
    if speech:
        speak_gtts(speech)
    if cmd:
        handle_command(cmd)


def handle_command(cmd: str):
    """
    LED_ON/LED_OFFを受け取りBLE書き込み
    """
    if LED_CLIENT is None:
        print("LED操作不可: デバイスが未接続です。")
        return
    if cmd == "LED_ON":
        mt = 1; red, green, blue = 2, 8, 32
        duration, on_time, off_time, pattern = 5000, 1000, 500, 1
    elif cmd == "LED_OFF":
        mt = 1; red = green = blue = 0
        duration = on_time = off_time = 0; pattern = 1
    else:
        print(f"Unknown command: {cmd}")
        return
    payload = pack('<BBBBBBBHHHB', mt, 0,
                   red, 0, green, 0, blue,
                   duration, on_time, off_time, pattern)
    checksum = sum(payload) & 0xFF
    payload += pack('B', checksum)
    try:
        LOOP.run_until_complete(
            LED_CLIENT.write_gatt_char(CORE_WRITE_UUID, payload, response=True)
        )
        print(f"→ 実行: {cmd}")
    except Exception as e:
        print("LED制御エラー:", e)


def chat_gemini(prompt: str) -> str:
    """Gemini APIのチャット機能で対話を実行。例外時にフォールバック応答を返す"""
    try:
        response = chat.send_message(prompt)
        return response.text
    except ServerError as e:
        print(f"Gemini API error: {e}")
        return json.dumps({
            "speech": "すみません、システムが混雑しています。後ほどお試しください。",
            "command": ""
        })


def recognize_until_endpoint():
    """
    発話開始～終了までマイク入力をストリーミング認識
    ポートオーディオエラー発生時は再試行
    """
    recognizer.Reset()
    frame_size = int(SAMPLE_RATE * 0.125)
    while True:
        try:
            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=frame_size,
                dtype="int16", channels=1
            ) as stream:
                print("▶ 録音開始…（話し始めてください）")
                while True:
                    data, _ = stream.read(frame_size)
                    audio_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
                    if recognizer.AcceptWaveform(audio_bytes):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        print(f"■ 発話終了検知 → {text}")
                        return text
        except sd.PortAudioError as e:
            print(f"Audio Input Error: {e}. 再試行します...")
            time.sleep(1)
        except Exception as e:
            print(f"音声認識中の予期せぬエラー: {e}")
            return ""


def main():
    """
    メインループ: LED接続→初期応答→対話ループ
    """
    global LED_CLIENT
    print("=== 音声対話エージェント + LED制御 ===")
    print("→ LED デバイスに接続中…")
    LED_CLIENT = LOOP.run_until_complete(connect_led())
    if LED_CLIENT is None:
        print("LED デバイス未検出: 対話のみ行います。")
    else:
        print("→ LED デバイス接続完了")
    initial_json = chat_gemini("")
    print(f"エージェント (raw): {initial_json}")
    parse_and_dispatch(initial_json)
    while True:
        user_text = recognize_until_endpoint()
        if not user_text:
            continue
        if user_text.lower() in ("終了", "やめて", "exit", "quit"):
            print("=== 終了します ===")
            break
        print(f"ユーザー: {user_text}")
        reply = chat_gemini(user_text)
        print(f"エージェント (raw): {reply}")
        parse_and_dispatch(reply)

if __name__ == "__main__":
    main()
