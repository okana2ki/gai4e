import os
import json
import tempfile

import sounddevice as sd
from vosk import Model, KaldiRecognizer
# from google.generativeai import Client
from google import genai
from gtts import gTTS
# from playsound import playsound
try:
    from playsound3 import playsound
except ImportError:
    # playsound3 が入っていなければ従来版へフォールバック
    from playsound import playsound

# ── 設定 ──
# スクリプトファイル自身の場所を基準にモデルフォルダを組み立て
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-ja-0.22")  # Vosk日本語モデルのパス：自分の環境に合わせて書き換え
# 解凍したディレクトリ名をそのまま使っているなら、"vosk-model-small-ja-0.22"とかにする
# MODEL_PATH      = "./model-ja"     # Vosk日本語モデルのパス：自分の環境に合わせて書き換え
# MODEL_PATH      = "C:/Briefcase/__python/voice_agent/model-ja"     # Vosk日本語モデルのパス：自分の環境に合わせて書き換え
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
        # contents=prompt
        contents=prompt,
        config={
            "maxOutputTokens": 50,       # ここを小さめにして応答があまり長くならないように
            "candidateCount": 1,         # 応答候補は１つ
            "temperature": 0.5,          # 必要に応じて応答の“ばらつき”も抑制
        }        
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