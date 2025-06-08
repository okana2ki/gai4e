#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# voice_agent.py
# 音声対話エージェント: PDFファイルからワークショップ情報を読み込み、
# その内容に基づいてワークショップへの参加を勧める対話エージェント

import os
import json
import tempfile

import sounddevice as sd
from vosk import Model, KaldiRecognizer
from google import genai
from google.genai import types
from gtts import gTTS
import httpx
try:
    from playsound3 import playsound
except ImportError:
    from playsound import playsound  # playsound3 が入っていなければ従来版へフォールバック

# ── Vosk/SampleRate 設定 ──
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-ja-0.22")
# Vosk日本語モデルのパス：自分の環境に合わせて書き換え；smallモデルを使っていて、解凍後ファイル名を変えてなければこのままでOK

# デフォルト入力デバイスのサンプルレートを自動取得し、失敗時は16000Hzをフォールバック
try:
    default_input_device = sd.default.device[0]
    device_info = sd.query_devices(default_input_device, 'input')
    SAMPLE_RATE = int(device_info['default_samplerate'])
    print(f"使用マイク ({device_info['name']}) のデフォルトサンプルレート: {SAMPLE_RATE} Hz")
    # AUDIO_DEVICEの設定を修正
    AUDIO_DEVICE = default_input_device if isinstance(default_input_device, int) and default_input_device >= 0 else None
except Exception as e:
    print(f"入力デバイス情報の取得に失敗: {e}")
    print("16000Hz を使用します。")
    SAMPLE_RATE = 16000
    AUDIO_DEVICE = None

# ── 初期化 ──
model      = Model(MODEL_PATH)  # Vosk モデルロード
recognizer = KaldiRecognizer(model, SAMPLE_RATE)
client     = genai.Client()     # 環境変数から API Key 自動取得


def load_pdf_content(pdf_url_or_path):
    """
    PDFファイルを読み込んで内容をテキストとして取得する
    URLまたはローカルパスに対応
    """
    try:
        if pdf_url_or_path.startswith(('http://', 'https://')):
            # URL からPDFを取得
            print(f"PDFをダウンロード中: {pdf_url_or_path}")
            doc_data = httpx.get(pdf_url_or_path, timeout=30).content
        else:
            # ローカルファイルから読み込み
            print(f"ローカルPDFを読み込み中: {pdf_url_or_path}")
            with open(pdf_url_or_path, 'rb') as f:
                doc_data = f.read()
        
        # PDFの内容を要約してテキスト化
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(
                    data=doc_data,
                    mime_type='application/pdf',
                ),
                "このPDFの内容を詳しく要約してください。特にワークショップの内容、日時、場所、参加方法、参加メリットなどの情報を含めてください。"
            ]
        )
        return response.text
    except httpx.TimeoutException:
        print("PDF読み込みエラー: タイムアウトしました")
        return None
    except Exception as e:
        print(f"PDF読み込みエラー: {e}")
        return None


def create_system_instruction(pdf_content):
    """
    PDFの内容に基づいてシステムインストラクションを生成する
    """
    if pdf_content:
        system_instruction = f"""
あなたは「みんなの生成AIワークショップ」の案内担当者です。以下のワークショップへの参加を積極的に勧めてください。
相手の興味や関心に合わせて、このワークショップの魅力や参加メリットを伝えてください。
ワークショップの最新情報については、ネットで「みんなの生成AIワークショップ」を検索して確認できると伝えてください。
音声対話向けの短めで親しみやすい応答を心がけてください。
音声対話なので、絵文字や記号（*、#、等）は使用しないでください。

【ワークショップ情報】
{pdf_content}

ユーザの質問に答えたり、参加を迷っている理由を聞いて適切にアドバイスしたりしてください。
最終的に参加申込みを促すことが目標ですが、押し付けがましくならないよう注意してください。
音声対話向けの短めで親しみやすい応答を心がける。
絵文字や記号は出力しない。
尋ねられた情報がない場合は、「その情報はないので、ネットで「みんなの生成AIワークショップ」を検索して下さい」と伝える。
"""
    else:
        system_instruction = """
あなたは「みんなの生成AIワークショップ」の案内担当者です。
ワークショップの内容について詳細な情報は現在取得できませんが、
ユーザの興味や関心を聞き、一般的なワークショップ参加のメリットを伝えて下さい。
ワークショップの詳細な情報については、ネットで「みんなの生成AIワークショップ」を検索して確認できると伝えてください。
音声対話向けの短めで親しみやすい応答を心がけてください。
絵文字や記号は使用しないでください。
"""
    
    return system_instruction


def speak_gtts(text, lang="ja"):
    """
    gTTS でテキストを音声合成し、一時ファイル経由で再生、完了を待つ
    再生完了後にファイルを削除
    """
    try:
        # 記号や絵文字を除去（音声合成に不適切な文字を除去）
        clean_text = ''.join(char for char in text if char.isprintable() and ord(char) < 127 or char.isspace() or ord(char) > 127)
        clean_text = clean_text.replace('*', '').replace('#', '').replace('✓', '').replace('⚠', '')
        
        if not clean_text.strip():
            print("⚠ 音声合成対象のテキストが空です")
            return
            
        tts = gTTS(text=clean_text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tts.save(f.name)
            path = f.name
        playsound(path)
        os.remove(path)
    except Exception as e:
        print(f"音声合成エラー: {e}")


def recognize_until_endpoint():
    """
    マイク入力を受け取り、発話開始から発話終了まで（Vosk の AcceptWaveform が True を返すまで）
    ストリーミング認識を継続。発話終了時点で認識結果のテキストを返す。
    """
    recognizer.Reset()  # 前回のセッションをリセット
    CHUNK_DURATION = 0.125  # 秒 (125ms)
    FRAME_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

    # RawInputStream で PCM 16bit モノラルデータを取得
    try:
        stream_params = {
            'samplerate': SAMPLE_RATE,
            'blocksize': FRAME_SIZE,
            'dtype': 'int16',
            'channels': 1,
        }
        
        # デバイスが指定されている場合のみ追加
        if AUDIO_DEVICE is not None:
            stream_params['device'] = AUDIO_DEVICE
            
        with sd.RawInputStream(**stream_params) as stream:
            print("▶ 録音開始…（話し始めてください）")
            silence_count = 0
            max_silence = 100  # 無音検出のカウンタ上限
            
            while True:
                try:
                    data, _ = stream.read(FRAME_SIZE)
                    # Vosk は bytes が必要 (_cffi_backend.buffer を bytes() で変換)
                    audio_bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
                    
                    # エンドポイント検知 (発話終了)
                    if recognizer.AcceptWaveform(audio_bytes):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").strip()
                        if text:
                            print("■ 発話終了検知 →", text)
                            return text
                    
                    # 部分認識結果の取得（デバッグ用）
                    partial_result = json.loads(recognizer.PartialResult())
                    partial_text = partial_result.get("partial", "")
                    if partial_text:
                        print(f"認識中: {partial_text}", end='\r')
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count > max_silence:
                            print("\n⚠ 長時間無音のため録音を終了します")
                            return ""
                            
                except KeyboardInterrupt:
                    print("\n録音を中断しました")
                    return ""
                except Exception as e:
                    print(f"\n録音中にエラー: {e}")
                    return ""
                    
    except Exception as e:
        print(f"音声入力ストリームの開始に失敗: {e}")
        print("キーボード入力に切り替えます。")
        return input("テキストで入力してください: ")


def chat_gemini(prompt, chat_session):
    """
    Gemini API のチャット機能を使って対話を実行。
    履歴を保持した session に user prompt を送信し、
    応答テキストを返す。
    """
    try:
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API エラー: {e}")
        return "申し訳ございません。システムにエラーが発生しました。もう一度お試しください。"


def main():
    """
    メインループ: PDFを読み込み、その内容に基づくシステムインストラクションを設定し、
    ワークショップ参加を勧める音声対話を実行
    """
    print("=== ワークショップ案内音声エージェント ===")
    
    # PDFファイルのパスまたはURL（必要に応じて変更してください）
    pdf_source = "https://okana2ki.github.io/gai4e-ws.pdf"
    # pdf_source = "workshop_info.pdf"  # ローカルファイルの場合
    
    print("PDFからワークショップ情報を読み込み中...")
    pdf_content = load_pdf_content(pdf_source)
    
    if pdf_content:
        print("✓ PDFの読み込みが完了しました")
        print("--- PDFの内容 ---")
        print(pdf_content[:500] + "..." if len(pdf_content) > 500 else pdf_content)
        print("--- 内容ここまで ---")
    else:
        print("⚠ PDFの読み込みに失敗しました。一般的な案内で進めます。")
    
    # システムインストラクションを生成
    system_instruction = create_system_instruction(pdf_content)
    
    try:
        # チャット機能（対話履歴を保持したマルチターン会話）の初期設定
        chat = client.chats.create(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=120,    # ワークショップ案内のため少し長めに設定
            )
        )
        
        # 初回システム発話: 空文字で system_instruction に応じた応答を取得
        initial_response = chat.send_message("こんにちは")
        initial_text = initial_response.text
        print("エージェント:", initial_text)
        speak_gtts(initial_text)

        while True:
            # ユーザーの音声を認識 (発話終了まで待機)
            user_text = recognize_until_endpoint()
            if not user_text.strip():
                print("⚠ 音声が認識できませんでした。もう一度お話しください。")
                continue
                
            # 終了キーワードチェック
            if any(keyword in user_text.lower() for keyword in ("終了", "やめて", "さようなら", "終わり")):
                farewell = "ありがとうございました。ワークショップでお会いできることを楽しみにしています！"
                print("エージェント:", farewell)
                speak_gtts(farewell)
                print("=== 終了します ===")
                break

            print("ユーザー:", user_text)
            # LLM へ問い合わせ
            reply = chat_gemini(user_text, chat)
            print("エージェント:", reply)
            # 応答を音声合成して再生
            speak_gtts(reply)
            
    except KeyboardInterrupt:
        print("\n=== プログラムを終了します ===")
    except Exception as e:
        print(f"プログラムエラー: {e}")


if __name__ == "__main__":
    main()