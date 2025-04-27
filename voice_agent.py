import os
import json
import tempfile

import sounddevice as sd
from vosk import Model, KaldiRecognizer
# from google.generativeai import Client
from google import genai
from gtts import gTTS
from playsound import playsound
# from playsound2 import playsound


# ── 設定 ──
# MODEL_PATH      = "./vosk-model-small-ja-0.22"     # Vosk日本語モデルのパス
MODEL_PATH      = "C:/Briefcase/__python/voice_agent/model-ja"     # Vosk日本語モデルのパス
SAMPLE_RATE     = 16000
RECORD_SECONDS  = 5                # 固定録音時間（秒）

# ── 初期化 ──
model      = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
client     = genai.Client()  # GOOGLE_API_KEY を環境変数から自動取得

def speak_gtts(text, lang="ja"):
    """gTTS で音声を合成＆再生し、完了を待つ"""
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        tts.save(f.name)
        path = f.name
    playsound(path)
    os.remove(path)

def record_fixed(duration):
    """固定秒数だけマイク録音し、NumPy array を返す"""
    audio = sd.rec(int(duration * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio

def recognize_vosk(audio):
    """Vosk で認識し、結果テキストを返す"""
    data = audio.tobytes()
    if recognizer.AcceptWaveform(data):
        res = recognizer.Result()
    else:
        res = recognizer.FinalResult()
    return json.loads(res).get("text", "")

def chat_gemini(prompt):
    """Gemini API で応答生成"""
    # resp = client.generate_message(
        # model="models/text-bison-001",
        # prompt=prompt
    # )
    resp = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    # return resp["candidates"][0]["message"]
    return resp.text

def main():
    print("=== 音声対話エージェント ===")
    while True:
        # １）システム発話
        system_utt = "何か聞きたいことはありますか？"
        speak_gtts(system_utt)

        # ２）再生完了直後に録音開始
        print("▶ 録音開始...")
        audio = record_fixed(RECORD_SECONDS)
        print("■ 録音終了 → 認識中...")

        # ３）STT
        user_text = recognize_vosk(audio)
        print("ユーザー:", user_text)

        # 終了条件
        if user_text in ("終了", "やめて", "exit", "quit"):
            print("=== 終了します ===")
            break

        # ４）LLM 応答生成
        reply = chat_gemini(user_text)
        print("エージェント:", reply)

        # ５）TTS
        speak_gtts(reply)

if __name__ == "__main__":
    main()