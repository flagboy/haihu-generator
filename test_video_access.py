#!/usr/bin/env python3
"""
ビデオアクセスのテストスクリプト
"""

from pathlib import Path

import cv2

# テストする動画パス
video_paths = [
    "web_interface/uploads/output.mp4",
    "web_interface/uploads/20250606_163211_output.mp4",
    "/Users/flagboy/Works/haihu-generator/web_interface/uploads/output.mp4",
    "uploads/output.mp4",
]

print("ビデオアクセステスト開始")
print("=" * 50)

for video_path in video_paths:
    print(f"\nテスト: {video_path}")
    path_obj = Path(video_path)
    print(f"  絶対パス: {path_obj.absolute()}")
    print(f"  ファイル存在: {path_obj.exists()}")

    if path_obj.exists():
        # OpenCVで開く
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            print("  ✅ OpenCVで開けました")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  フレーム数: {frame_count}")
            print(f"  FPS: {fps}")

            # フレーム0を読む
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                print(f"  ✅ フレーム0を読めました (shape: {frame.shape})")
            else:
                print("  ❌ フレーム0を読めませんでした")

            cap.release()
        else:
            print("  ❌ OpenCVで開けませんでした")
    else:
        print("  ❌ ファイルが存在しません")

print("\n" + "=" * 50)
print("テスト完了")
