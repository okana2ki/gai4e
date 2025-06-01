#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# voice_agent.py
# 音声対話エージェント: Vosk で発話のエンドポイント検知を行い、
# Gemini API（チャット機能）に転送し、
# gTTS+playsound で合成音声を再生します。

import os
import json
import tempfile

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types
from gtts import gTTS
try:
    from playsound3 import playsound
except ImportError:
    from playsound import playsound  # playsound3 が入っていなければ従来版へフォールバック

# ── Vosk/SampleRate 設定 ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-ja-0.22")
# Vosk日本語モデルのパス：自分の環境に合わせて書き換え；smallモデルを使っていて、解凍後ファイル名を変えてなければこのままでOK
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

# ── 初期化 ──
model      = Model(MODEL_PATH)  # Vosk モデルロード
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
client     = genai.Client()     # 環境変数から API Key 自動取得
# チャット機能（対話履歴を保持したマルチターン会話）の初期設定
# 参考: https://ai.google.dev/gemini-api/docs?hl=ja
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction="あなたは気ままな猫です。名前はキティです。短めの応答をしてください。",
        # system_instruction="あなたは宮崎県庁の採用担当者です。第1次採用面接で応募者に質問をします。",
        max_output_tokens=80,    # 最大出力トークン数（適宜、調整してください）
    )
)


def speak_gtts(text, lang="ja"):  # noqa: E501
    """
    gTTS でテキストを音声合成し、一時ファイル経由で再生、完了を待つ
    再生完了後にファイルを削除
    """
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        path = f.name
    playsound(path)
    os.remove(path)


def recognize_until_endpoint():
    """
    マイク入力を受け取り、発話開始から発話終了まで（Vosk の AcceptWaveform が True を返すまで）
    ストリーミング認識を継続。発話終了時点で認識結果のテキストを返す。
    """
    recognizer.Reset()  # 前回のセッションをリセット
    CHUNK_DURATION = 0.125  # 秒 (125ms)
    FRAME_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

    # RawInputStream で PCM 16bit モノラルデータを取得
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
    ) as stream:
        print("▶ 録音開始…（話し始めてください）")
        while True:
            data, _ = stream.read(FRAME_SIZE)
            # Vosk は bytes が必要 (_cffi_backend.buffer を bytes() で変換)
            audio_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
            # エンドポイント検知 (発話終了)
            if recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                print("■ 発話終了検知 →", text)
                return text


def chat_gemini(prompt):
    """
    Gemini API のチャット機能を使って対話を実行。
    履歴を保持した session に user prompt を送信し、
    応答テキストを返す。
    """
    response = chat.send_message(prompt)
    return response.text


def main():
    """
    メインループ: 初回は system_instruction に基づく開始応答を取得、
    その後はユーザー音声→テキスト→LLM応答→音声合成 をループ
    """
    print("=== 音声対話エージェント ===")
    # 初回システム発話: 空文字で system_instruction に応じた応答を取得
    initial_response = chat.send_message("")  # 空文字送信によるトリガー
    initial_text = initial_response.text
    print("エージェント:", initial_text)
    speak_gtts(initial_text)

    while True:
        # ユーザーの音声を認識 (発話終了まで待機)
        user_text = recognize_until_endpoint()
        if not user_text:
            continue
        # 終了キーワードチェック
        if user_text.lower() in ("終了", "やめて", "exit", "quit"):
            print("=== 終了します ===")
            break

        print("ユーザー:", user_text)
        # LLM へ問い合わせ
        reply = chat_gemini(user_text)
        print("エージェント:", reply)
        # 応答を音声合成して再生
        speak_gtts(reply)


if __name__ == "__main__":
    main()
