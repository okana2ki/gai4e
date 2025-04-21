import asyncio
from bleak import BleakClient, BleakScanner
from struct import pack

# MESH モジュールの UUID 定義
CORE_NOTIFY_UUID   = '72c90003-57a9-4d40-b746-534e22ec9f9e'
CORE_INDICATE_UUID = '72c90005-57a9-4d40-b746-534e22ec9f9e'
CORE_WRITE_UUID    = '72c90004-57a9-4d40-b746-534e22ec9f9e'

# ボタン通知データのインデックス／ID
MESSAGE_TYPE_INDEX = 0
EVENT_TYPE_INDEX   = 1
STATE_INDEX        = 2
MESSAGE_TYPE_ID    = 1
EVENT_TYPE_ID      = 0

async def scan(prefix: str):
    """名前が prefix で始まるデバイスをスキャンして返す"""
    while True:
        for d in await BleakScanner.discover():
            if d.name and d.name.startswith(prefix):
                return d
        await asyncio.sleep(1)

async def main():
    print("Scanning for BUTTON module...")
    button_dev = await scan('MESH-100BU')
    print(f"Found BUTTON: {button_dev.name} [{button_dev.address}]")

    print("Scanning for LED module...")
    led_dev = await scan('MESH-100LE')
    print(f"Found LED:    {led_dev.name} [{led_dev.address}]")

    # 同時に２つのクライアントを接続
    async with BleakClient(button_dev, timeout=None) as client_btn, \
               BleakClient(led_dev,    timeout=None) as client_led:

        # --- 初期化 ---
        # BUTTON モジュール
        await client_btn.start_notify(CORE_NOTIFY_UUID,   lambda s,d: None)
        await client_btn.start_notify(CORE_INDICATE_UUID, lambda s,d: None)
        await client_btn.write_gatt_char(CORE_WRITE_UUID, pack('<BBBB', 0,2,1,3), response=True)

        # LED モジュール
        await client_led.start_notify(CORE_NOTIFY_UUID,   lambda s,d: None)
        await client_led.start_notify(CORE_INDICATE_UUID, lambda s,d: None)
        await client_led.write_gatt_char(CORE_WRITE_UUID, pack('<BBBB', 0,2,1,3), response=True)

        # --- LED 点灯コマンド作成 ---
        messagetype = 1
        red, green, blue = 2, 8, 32
        duration = 5 * 1000   # 5,000 ms
        on_time  = 1 * 1000   # 1,000 ms
        off_time = 500        # 500 ms
        pattern  = 1          # 1=blink
        cmd = pack('<BBBBBBBHHHB',
                   messagetype, 0,
                   red, 0, green, 0, blue,
                   duration, on_time, off_time, pattern)
        checksum = sum(cmd) & 0xFF
        led_command = cmd + pack('B', checksum)

        # --- ボタン押下コールバック ---
        def on_button_pressed(sender: int, data: bytearray):
            if (data[MESSAGE_TYPE_INDEX] == MESSAGE_TYPE_ID and
                data[EVENT_TYPE_INDEX]   == EVENT_TYPE_ID and
                data[STATE_INDEX]        == 1):
                print("Button single press detected → Lighting LED")
                # 非同期タスクで LED コマンド送信
                asyncio.create_task(
                    client_led.write_gatt_char(CORE_WRITE_UUID, led_command, response=True)
                )

        # 通知コールバックを差し替え
        await client_btn.stop_notify(CORE_NOTIFY_UUID)
        await client_btn.start_notify(CORE_NOTIFY_UUID, on_button_pressed)

        print("Ready: ボタンを押すと LED が点灯します。30秒後に終了。")
        await asyncio.sleep(30)

if __name__ == '__main__':
    asyncio.run(main())