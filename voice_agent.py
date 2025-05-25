import os
import json
import tempfile

import sounddevice as sd
from vosk import Model, KaldiRecognizer
# from google.generativeai import Client
from google import genai
from google.genai import types
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
# チャット機能（対話履歴を使ったマルチターンの会話）の初期設定
# 参考: https://ai.google.dev/gemini-api/docs?hl=ja
chat = client.chats.create(
    model="gemini-2.0-flash",
    # model="gemini-2.5-flash-preview-05-20",
    # model="gemini-2.0-flash-001",
    config=types.GenerateContentConfig(
        system_instruction="あなたは気ままな猫です。名前はキティです。",  # システムインストラクション
        max_output_tokens=50,  # 最大出力トークン数
    )
)   

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

# def chat_gemini(prompt):
#     response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         config=types.GenerateContentConfig(
#             system_instruction="You are a cat. Your name is Neko.",
#             max_output_tokens=50,
#             ),
#         contents=prompt,
#     )
#     return response.text

def chat_gemini(prompt):
    """Gemini API を使ってチャット応答を生成"""
    response = chat.send_message(prompt)
    return response.text

# 参考: Gemini API のチャット機能を使う場合 https://ai.google.dev/gemini-api/docs?hl=ja
# chat = client.chats.create(model="gemini-2.0-flash")
# response = chat.send_message("I have 2 dogs in my house.")
# print(response.text)
# response = chat.send_message("How many paws are in my house?")
# print(response.text)

def main(): 
    print("=== 音声対話エージェント ===")
    # system_utt = "何か聞きたいことはありますか？"
    system_utt = "何か聞きたいことある？"
    speak_gtts(system_utt)
    while True:
        # １）システム発話
        # system_utt = "何か聞きたいことはありますか？"
        # speak_gtts(system_utt)

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