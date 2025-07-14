import asyncio
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from collections import deque
import numpy as np
from bleak import BleakClient, BleakScanner
from struct import pack, unpack
import time
from datetime import datetime
import threading
import os
import warnings
import sys

# 警告を抑制
warnings.filterwarnings('ignore')
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# 日本語フォント設定（japanize_matplotlibを使用）
try:
    import japanize_matplotlib
    print("✅ japanize_matplotlib を使用して日本語フォントを設定しました")
except ImportError:
    print("⚠️ japanize_matplotlib がインストールされていません")
    print("   pip install japanize-matplotlib でインストールしてください")
    # フォールバック設定
    import matplotlib
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# UUID (MESHブロック共通)
CORE_INDICATE_UUID = ('72c90005-57a9-4d40-b746-534e22ec9f9e')
CORE_NOTIFY_UUID = ('72c90003-57a9-4d40-b746-534e22ec9f9e')
CORE_WRITE_UUID = ('72c90004-57a9-4d40-b746-534e22ec9f9e')

# 定数値
MESSAGE_TYPE_ID = 0x01
EVENT_TYPE_TAP = 0x00
EVENT_TYPE_SHAKE = 0x01
EVENT_TYPE_FLIP = 0x02
EVENT_TYPE_ORIENTATION = 0x03

class DataVisualizer:
    def __init__(self, max_points=300):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.x_data = deque(maxlen=max_points)
        self.y_data = deque(maxlen=max_points)
        self.z_data = deque(maxlen=max_points)
        self.total_data = deque(maxlen=max_points)
        
        # ジャンプ検出状態の記録
        self.jump_events = []
        self.jump_phases = deque(maxlen=max_points)
        self.jump_states = deque(maxlen=max_points)
        
        # 自動保存用の設定
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存場所を指定
        base_dir = "C:/Briefcase/__python/mesh"
        
        # ディレクトリが存在しない場合は作成
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            print(f"⚠️ 指定ディレクトリの作成に失敗: {e}")
            print("   カレントディレクトリに保存します")
            base_dir = os.getcwd()
        
        self.save_dir = os.path.join(base_dir, f"mesh_data_{self.session_id}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.last_save_time = time.time()
        self.save_interval = 30  # 30秒ごとに自動保存
        
        # グラフ設定
        plt.style.use('default')  # デフォルトスタイルを使用
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 日本語タイトル（japanize_matplotlibがあれば確実に表示）
        self.fig.suptitle('MESH動きブロック 加速度データ リアルタイム表示', fontsize=14, fontweight='bold')
        
        # 線の初期化
        self.line_x, = self.ax1.plot([], [], 'r-', label='X軸', linewidth=1.5)
        self.line_y, = self.ax1.plot([], [], 'g-', label='Y軸', linewidth=1.5)
        self.line_z, = self.ax1.plot([], [], 'b-', label='Z軸', linewidth=1.5)
        self.line_total, = self.ax2.plot([], [], 'purple', label='合成加速度', linewidth=2)
        
        # 閾値線
        self.ax2.axhline(y=1.3, color='red', linestyle='--', alpha=0.7, label='ジャンプ開始閾値(1.3G)')
        self.ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.7, label='自由落下閾値(0.7G)')
        self.ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='重力(1.0G)')
        
        # 軸設定（日本語表示）
        self.ax1.set_ylabel('加速度 (G)', fontsize=10)
        self.ax1.set_title('3軸加速度', fontweight='bold', fontsize=12)
        self.ax2.set_ylabel('合成加速度 (G)', fontsize=10)
        self.ax2.set_title('合成加速度とジャンプ検出', fontweight='bold', fontsize=12)
        self.ax3.set_ylabel('ジャンプ状態', fontsize=10)
        self.ax3.set_xlabel('時間 (秒)', fontsize=10)
        self.ax3.set_title('ジャンプ検出状態', fontweight='bold', fontsize=12)
        
        self.ax1.legend(loc='upper right', fontsize=9)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-3, 3)
        
        self.ax2.legend(loc='upper right', fontsize=9)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 3)
        
        # ジャンプ状態表示用のサブプロット
        self.ax3.set_ylim(-0.5, 3.5)
        self.ax3.set_yticks([0, 1, 2, 3])
        self.ax3.set_yticklabels(['待機', '離陸', '空中', '着地'], fontsize=9)
        self.ax3.grid(True, alpha=0.3)
        
        # 開始時刻
        self.start_time = time.time()
        
        # データロック
        self.data_lock = threading.Lock()
        
        print(f"📊 データ保存ディレクトリ: {self.save_dir}")
        
    def add_data(self, x_g, y_g, z_g, jump_detector):
        """新しいデータを追加"""
        with self.data_lock:
            current_time = time.time() - self.start_time
            
            self.times.append(current_time)
            self.x_data.append(x_g)
            self.y_data.append(y_g)
            self.z_data.append(z_g)
            
            total_g = math.sqrt(x_g**2 + y_g**2 + z_g**2)
            self.total_data.append(total_g)
            
            # ジャンプ状態を記録
            if jump_detector.is_jumping:
                if jump_detector.jump_phase == 'takeoff':
                    phase_num = 1
                elif jump_detector.jump_phase == 'airborne':
                    phase_num = 2
                elif jump_detector.jump_phase == 'landing':
                    phase_num = 3
                else:
                    phase_num = 1
            else:
                phase_num = 0
                
            self.jump_phases.append(phase_num)
            self.jump_states.append(1 if jump_detector.is_jumping else 0)
            
            # 定期的な自動保存
            if time.time() - self.last_save_time > self.save_interval:
                self.auto_save_data()
                self.last_save_time = time.time()
    
    def add_jump_event(self, event_type, timestamp, details):
        """ジャンプイベントを記録"""
        relative_time = timestamp - self.start_time
        self.jump_events.append({
            'type': event_type,
            'time': relative_time,
            'details': details
        })
    
    def auto_save_data(self):
        """データの自動保存"""
        try:
            # CSVデータ保存
            if len(self.times) > 0:
                data_file = os.path.join(self.save_dir, f"acceleration_data_{datetime.now().strftime('%H%M%S')}.csv")
                with open(data_file, 'w', encoding='utf-8') as f:
                    f.write("Time,X_G,Y_G,Z_G,Total_G,Jump_Phase,Jump_State\n")
                    for i in range(len(self.times)):
                        f.write(f"{self.times[i]:.3f},{self.x_data[i]:.3f},{self.y_data[i]:.3f},"
                               f"{self.z_data[i]:.3f},{self.total_data[i]:.3f},"
                               f"{self.jump_phases[i]},{self.jump_states[i]}\n")
            
            # ジャンプイベント保存
            if self.jump_events:
                events_file = os.path.join(self.save_dir, f"jump_events_{datetime.now().strftime('%H%M%S')}.csv")
                with open(events_file, 'w', encoding='utf-8') as f:
                    f.write("Event_Type,Time,Duration,Height,Power,Max_Acc,Min_Acc\n")
                    for event in self.jump_events:
                        if event['type'] == 'complete':
                            details = event['details']
                            f.write(f"complete,{event['time']:.3f},{details.get('duration', 0):.3f},"
                                   f"{details.get('height', 0):.1f},{details.get('power', 0):.1f},"
                                   f"{details.get('max_acc', 0):.3f},{details.get('min_acc', 0):.3f}\n")
                        else:
                            f.write(f"{event['type']},{event['time']:.3f},,,,\n")
        except Exception as e:
            print(f"⚠️ 自動保存エラー: {e}")
    
    def save_final_graph(self):
        """最終グラフ保存"""
        try:
            # PNGとして保存
            graph_file = os.path.join(self.save_dir, f"final_graph_{self.session_id}.png")
            self.fig.savefig(graph_file, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            
            # PDFとしても保存
            pdf_file = os.path.join(self.save_dir, f"final_graph_{self.session_id}.pdf")
            self.fig.savefig(pdf_file, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            
            print(f"💾 グラフを保存しました: {graph_file}")
            print(f"💾 PDFも保存しました: {pdf_file}")
            
            # 最終データ保存
            self.auto_save_data()
            
            # サマリー情報保存
            summary_file = os.path.join(self.save_dir, f"session_summary_{self.session_id}.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"MESH加速度測定セッション サマリー\n")
                f.write(f"セッションID: {self.session_id}\n")
                f.write(f"測定開始時刻: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"測定終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"総測定時間: {(time.time() - self.start_time):.1f}秒\n")
                f.write(f"データポイント数: {len(self.times)}\n")
                f.write(f"ジャンプイベント数: {len([e for e in self.jump_events if e['type'] == 'complete'])}\n")
                
                if self.jump_events:
                    completed_jumps = [e for e in self.jump_events if e['type'] == 'complete']
                    if completed_jumps:
                        heights = [e['details']['height'] for e in completed_jumps]
                        powers = [e['details']['power'] for e in completed_jumps]
                        f.write(f"最大ジャンプ高: {max(heights):.1f}cm\n")
                        f.write(f"平均ジャンプ高: {sum(heights)/len(heights):.1f}cm\n")
                        f.write(f"最大ジャンプ力: {max(powers):.1f}点\n")
                        f.write(f"平均ジャンプ力: {sum(powers)/len(powers):.1f}点\n")
            
            print(f"📋 セッションサマリーを保存しました: {summary_file}")
            
        except Exception as e:
            print(f"⚠️ グラフ保存エラー: {e}")
    
    def update_plot(self, frame):
        """グラフを更新"""
        with self.data_lock:
            if len(self.times) == 0:
                return self.line_x, self.line_y, self.line_z, self.line_total
            
            times_array = np.array(self.times)
            
            # 3軸加速度グラフ更新
            self.line_x.set_data(times_array, np.array(self.x_data))
            self.line_y.set_data(times_array, np.array(self.y_data))
            self.line_z.set_data(times_array, np.array(self.z_data))
            
            # 合成加速度グラフ更新
            self.line_total.set_data(times_array, np.array(self.total_data))
            
            # X軸の範囲を調整（最新30秒間を表示）
            if len(times_array) > 0:
                latest_time = times_array[-1]
                start_time = max(0, latest_time - 30)
                
                self.ax1.set_xlim(start_time, latest_time + 1)
                self.ax2.set_xlim(start_time, latest_time + 1)
                self.ax3.set_xlim(start_time, latest_time + 1)
            
            # ジャンプ状態の背景色を更新
            self.ax3.clear()
            self.ax3.set_ylabel('ジャンプ状態', fontsize=10)
            self.ax3.set_xlabel('時間 (秒)', fontsize=10)
            self.ax3.set_title('ジャンプ検出状態', fontweight='bold', fontsize=12)
            self.ax3.set_ylim(-0.5, 3.5)
            self.ax3.set_yticks([0, 1, 2, 3])
            self.ax3.set_yticklabels(['待機', '離陸', '空中', '着地'], fontsize=9)
            self.ax3.grid(True, alpha=0.3)
            
            if len(times_array) > 0:
                latest_time = times_array[-1]
                start_time = max(0, latest_time - 30)
                self.ax3.set_xlim(start_time, latest_time + 1)
                
                # ジャンプ状態を段階的に色分け表示
                phases_array = np.array(self.jump_phases)
                for i in range(len(times_array)):
                    if i > 0:
                        phase = phases_array[i]
                        color = ['lightgray', 'lightcoral', 'lightblue', 'lightgreen'][phase]
                        self.ax3.axvspan(times_array[i-1], times_array[i], 
                                       ymin=phase/4, ymax=(phase+1)/4, 
                                       color=color, alpha=0.7)
            
            # ジャンプイベントマーカーを追加
            for event in self.jump_events:
                if event['time'] >= start_time:
                    if event['type'] == 'start':
                        self.ax2.axvline(x=event['time'], color='green', 
                                       linestyle='-', linewidth=2, alpha=0.8)
                        self.ax2.text(event['time'], 2.5, '🚀開始', 
                                    rotation=90, fontsize=8, color='green')
                    elif event['type'] == 'complete':
                        self.ax2.axvline(x=event['time'], color='blue', 
                                       linestyle='-', linewidth=2, alpha=0.8)
                        details = event['details']
                        label = f"🎯完了\n{details['duration']:.1f}s\n{details['height']:.0f}cm"
                        self.ax2.text(event['time'], 2.2, label, 
                                    rotation=90, fontsize=7, color='blue')
        
        return self.line_x, self.line_y, self.line_z, self.line_total
    
    def start_animation(self):
        """アニメーション開始"""
        try:
            self.ani = animation.FuncAnimation(self.fig, self.update_plot, 
                                             interval=50, blit=False, cache_frame_data=False)
            plt.tight_layout()
            
            # 終了時の処理を設定
            def on_close(event):
                print("\n💾 グラフウィンドウが閉じられました。データを保存中...")
                self.save_final_graph()
            
            self.fig.canvas.mpl_connect('close_event', on_close)
            
            plt.show()
        except Exception as e:
            print(f"⚠️ グラフ表示エラー: {e}")
            # エラーが発生してもデータは保存
            self.save_final_graph()

class EnhancedJumpDetector:
    """可視化機能付きジャンプ検出器"""
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.is_jumping = False
        self.jump_start_time = None
        self.jump_start_acceleration = None
        self.max_acceleration = 0
        self.min_acceleration = float('inf')
        self.acceleration_history = []
        self.jump_phase = None
        self.phase_change_time = None
        self.jump_threshold_high = 1.3
        self.jump_threshold_low = 0.7
        self.stable_threshold = 0.15
        self.max_jump_duration = 3.0
        
    def process_acceleration(self, x_g, y_g, z_g):
        """加速度処理（可視化機能付き）"""
        total_acceleration = math.sqrt(x_g**2 + y_g**2 + z_g**2)
        current_time = time.time()
        
        # 可視化データを追加
        self.visualizer.add_data(x_g, y_g, z_g, self)
        
        # 履歴に追加
        self.acceleration_history.append((current_time, total_acceleration, x_g, y_g, z_g))
        if len(self.acceleration_history) > 15:
            self.acceleration_history.pop(0)
        
        # タイムアウト処理
        if self.is_jumping and (current_time - self.jump_start_time) > self.max_jump_duration:
            print("⚠️  ジャンプタイムアウト - 強制終了")
            self.force_complete_jump(current_time, x_g, y_g, z_g)
            return
        
        if not self.is_jumping:
            if self.detect_jump_start(total_acceleration):
                self.start_jump(current_time, total_acceleration, x_g, y_g, z_g)
        else:
            self.update_jump_phase(total_acceleration, current_time)
            self.update_jump_metrics(total_acceleration)
            
            if self.detect_landing(total_acceleration, current_time):
                self.complete_jump(current_time, x_g, y_g, z_g)
    
    def detect_jump_start(self, total_acceleration):
        """ジャンプ開始検出"""
        if total_acceleration > self.jump_threshold_high:
            if len(self.acceleration_history) >= 3:
                recent_accelerations = [acc for _, acc, _, _, _ in self.acceleration_history[-3:]]
                if max(recent_accelerations) - min(recent_accelerations) > 0.3:
                    return True
        return False
    
    def update_jump_phase(self, total_acceleration, current_time):
        """ジャンプ段階更新"""
        if self.jump_phase == 'takeoff':
            if total_acceleration < self.jump_threshold_low:
                self.jump_phase = 'airborne'
                self.phase_change_time = current_time
                print("   📡 空中段階に移行")
        elif self.jump_phase == 'airborne':
            if total_acceleration > 1.1:
                self.jump_phase = 'landing'
                self.phase_change_time = current_time
                print("   🎯 着地段階に移行")
    
    def detect_landing(self, total_acceleration, current_time):
        """着地検出"""
        min_jump_time = 0.3
        if (current_time - self.jump_start_time) < min_jump_time:
            return False
        
        time_since_start = current_time - self.jump_start_time
        if self.jump_phase not in ['landing'] and time_since_start < 1.0:
            return False
        
        if len(self.acceleration_history) >= 5:
            recent_accelerations = [acc for _, acc, _, _, _ in self.acceleration_history[-5:]]
            avg_acceleration = sum(recent_accelerations) / len(recent_accelerations)
            
            if 0.8 <= avg_acceleration <= 1.2:
                variations = [abs(acc - avg_acceleration) for acc in recent_accelerations]
                max_variation = max(variations)
                
                if max_variation < self.stable_threshold:
                    return True
        
        if time_since_start > 1.5 and 0.7 <= total_acceleration <= 1.3:
            return True
            
        return False
    
    def start_jump(self, current_time, total_acceleration, x_g, y_g, z_g):
        """ジャンプ開始処理"""
        self.is_jumping = True
        self.jump_start_time = current_time
        self.jump_start_acceleration = total_acceleration
        self.max_acceleration = total_acceleration
        self.min_acceleration = total_acceleration
        self.jump_phase = 'takeoff'
        self.phase_change_time = current_time
        
        # 可視化にイベント追加
        self.visualizer.add_jump_event('start', current_time, {'acceleration': total_acceleration})
        
        print(f"\n🚀 ジャンプ開始検出!")
        print(f"   時刻: {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        print(f"   開始加速度: {total_acceleration:.3f}G")
        print(f"   3軸値: X={x_g:.3f}G, Y={y_g:.3f}G, Z={z_g:.3f}G")
        
    def update_jump_metrics(self, total_acceleration):
        """ジャンプメトリクス更新"""
        if total_acceleration > self.max_acceleration:
            self.max_acceleration = total_acceleration
            
        if total_acceleration < self.min_acceleration:
            self.min_acceleration = total_acceleration
    
    def complete_jump(self, current_time, x_g, y_g, z_g):
        """ジャンプ完了処理"""
        jump_duration = current_time - self.jump_start_time
        jump_height = self.calculate_jump_height()
        jump_power = self.calculate_jump_power()
        
        # 可視化にイベント追加
        details = {
            'duration': jump_duration,
            'height': jump_height,
            'power': jump_power,
            'max_acc': self.max_acceleration,
            'min_acc': self.min_acceleration
        }
        self.visualizer.add_jump_event('complete', current_time, details)
        
        print(f"\n🎯 ジャンプ完了!")
        print(f"   継続時間: {jump_duration:.2f}秒")
        print(f"   推定高さ: {jump_height:.1f}cm")
        print(f"   ジャンプ力: {jump_power:.1f}点")
        print(f"   最大加速度: {self.max_acceleration:.3f}G")
        print(f"   最小加速度: {self.min_acceleration:.3f}G")
        print(f"   着地時3軸値: X={x_g:.3f}G, Y={y_g:.3f}G, Z={z_g:.3f}G")
        print(f"   段階: {self.jump_phase}")
        print("─" * 50)
        
        self.reset_jump_state()
    
    def force_complete_jump(self, current_time, x_g, y_g, z_g):
        """強制ジャンプ完了"""
        print(f"\n⚠️  強制ジャンプ完了（タイムアウト）")
        self.complete_jump(current_time, x_g, y_g, z_g)
    
    def calculate_jump_height(self):
        """ジャンプ高さ計算"""
        if self.max_acceleration <= 1.0:
            return 0
            
        excess_acceleration = (self.max_acceleration - 1.0) * 9.81
        
        airborne_factor = 1.0
        if self.min_acceleration < 0.5:
            airborne_factor = 1.3
            
        estimated_velocity = excess_acceleration * 0.1 * airborne_factor
        height_m = (estimated_velocity ** 2) / (2 * 9.81)
        height_cm = height_m * 100
        
        return min(max(height_cm, 0), 150)
    
    def calculate_jump_power(self):
        """ジャンプ力計算"""
        height_score = min(self.calculate_jump_height() * 0.6, 50)
        acceleration_score = min((self.max_acceleration - 1.0) * 20, 30)
        
        airborne_bonus = 0
        if self.min_acceleration < 0.6:
            airborne_bonus = 15
            
        phase_bonus = 0
        if self.jump_phase in ['landing', 'airborne']:
            phase_bonus = 5
            
        total_score = height_score + acceleration_score + airborne_bonus + phase_bonus
        return min(max(total_score, 0), 100)
    
    def reset_jump_state(self):
        """ジャンプ状態リセット"""
        self.is_jumping = False
        self.jump_start_time = None
        self.jump_start_acceleration = None
        self.max_acceleration = 0
        self.min_acceleration = float('inf')
        self.jump_phase = None
        self.phase_change_time = None

def convert_sensor_value_to_acceleration(sensor_value):
    """センサー値を加速度（G）に変換"""
    if sensor_value <= 2047:
        return sensor_value / 1024.0
    else:
        return (sensor_value - 65536) / 1024.0

# グローバル変数
visualizer = None
jump_detector = None

def on_receive_notify(sender, data: bytearray):
    """動きブロックからの通知処理"""
    global jump_detector
    
    try:
        if len(data) < 10:
            return
            
        if data[0] != MESSAGE_TYPE_ID:
            return
        
        event_type = data[1]
        
        # 加速度データを取得
        x_raw = unpack('<H', data[4:6])[0]
        y_raw = unpack('<H', data[6:8])[0]
        z_raw = unpack('<H', data[8:10])[0]
        
        # センサー値を加速度（G）に変換
        x_g = convert_sensor_value_to_acceleration(x_raw)
        y_g = convert_sensor_value_to_acceleration(y_raw)
        z_g = convert_sensor_value_to_acceleration(z_raw)
        
        total_g = math.sqrt(x_g**2 + y_g**2 + z_g**2)
        
        # イベント種別を表示
        event_names = {
            EVENT_TYPE_TAP: "タップ",
            EVENT_TYPE_SHAKE: "シェイク", 
            EVENT_TYPE_FLIP: "フリップ",
            EVENT_TYPE_ORIENTATION: "向き変更"
        }
        event_name = event_names.get(event_type, f"不明({event_type})")
        
        print(f"[{event_name}] X:{x_g:+.3f}G Y:{y_g:+.3f}G Z:{z_g:+.3f}G 合成:{total_g:.3f}G")
        
        # ジャンプ検出処理
        if jump_detector:
            jump_detector.process_acceleration(x_g, y_g, z_g)
        
    except Exception as e:
        print(f"データ処理エラー: {e}")
        print(f"受信データ: {data.hex()}")

def on_receive_indicate(sender, data: bytearray):
    """Indicateメッセージの処理"""
    print(f'[Indicate] {data.hex()}')

async def scan_motion_block():
    """動きブロック（MESH-100AC）をスキャン"""
    print("動きブロック（MESH-100AC）をスキャン中...")
    
    retry_count = 0
    max_retries = 10
    
    while retry_count < max_retries:
        try:
            devices = await BleakScanner.discover(timeout=5.0)
            for device in devices:
                if device.name and 'MESH-100AC' in device.name:
                    print(f"✅ 動きブロックを発見: {device.name}")
                    return device
                    
            retry_count += 1
            print(f"スキャン {retry_count}/{max_retries} - 動きブロックが見つかりません...")
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"スキャンエラー: {e}")
            retry_count += 1
            await asyncio.sleep(2)
    
    raise Exception("動きブロックが見つかりませんでした。電源とペアリング状態を確認してください。")

async def bluetooth_main():
    """Bluetooth接続とデータ取得のメイン処理"""
    global jump_detector
    
    try:
        # 動きブロックをスキャン
        device = await scan_motion_block()
        
        # デバイスに接続
        print(f"\n📱 接続中: {device.name} ({device.address})")
        
        async with BleakClient(device, timeout=30.0) as client:
            print("✅ 接続成功!")
            
            # 通知を開始
            await client.start_notify(CORE_NOTIFY_UUID, on_receive_notify)
            await client.start_notify(CORE_INDICATE_UUID, on_receive_indicate)
            
            # 動きブロックの初期化
            init_command = pack('<BBBB', 0, 2, 1, 3)
            await client.write_gatt_char(CORE_WRITE_UUID, init_command, response=True)
            
            print("\n🎯 動きブロック準備完了!")
            print("💡 リアルタイムグラフが表示されます")
            print("   - 3軸加速度、合成加速度、ジャンプ状態が可視化されます")
            print("   - ジャンプするとグラフ上にマーカーが表示されます")
            print("   - データは自動的に保存されます")
            print("   - グラフウィンドウを閉じるか Ctrl+C で終了")
            print("─" * 50)
            
            # データ受信を継続
            try:
                while True:
                    await asyncio.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n\n👋 プログラムを終了します...")
                
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("\n🔧 トラブルシューティング:")
        print("1. 動きブロックの電源が入っているか確認")
        print("2. スマートフォンアプリでペアリングを解除") 
        print("3. Bluetoothが有効になっているか確認")

def signal_handler(signum, frame):
    """シグナルハンドラー（Ctrl+C対応）"""
    global visualizer
    print("\n\n💾 終了処理中...")
    if visualizer:
        visualizer.save_final_graph()
    print("👋 プログラムを終了しました")
    sys.exit(0)

def main():
    """メイン関数"""
    global visualizer, jump_detector
    
    # シグナルハンドラーを設定
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 MESH動きブロック ジャンプ検出・可視化システム v2.0")
    print("=" * 60)
    
    # 可視化システムを初期化
    visualizer = DataVisualizer()
    jump_detector = EnhancedJumpDetector(visualizer)
    
    # Bluetoothタスクを別スレッドで実行
    def run_bluetooth():
        try:
            asyncio.run(bluetooth_main())
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Bluetoothスレッドエラー: {e}")
    
    bluetooth_thread = threading.Thread(target=run_bluetooth, daemon=True)
    bluetooth_thread.start()
    
    # グラフアニメーション開始（メインスレッド）
    try:
        print("📊 グラフウィンドウを初期化中...")
        visualizer.start_animation()
    except KeyboardInterrupt:
        print('\n👋 プログラムを終了しました')
    except Exception as e:
        print(f"グラフ表示エラー: {e}")
    finally:
        # 確実にデータを保存
        if visualizer:
            visualizer.save_final_graph()

if __name__ == '__main__':
    main()