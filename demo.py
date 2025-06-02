#!/usr/bin/env python3
"""
麻雀牌譜作成システム デモスクリプト

基本的な動画処理機能をテストするためのデモ
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.classification.tile_classifier import TileClassifier
from src.detection.tile_detector import TileDetector
from src.models.model_manager import ModelManager
from src.pipeline.ai_pipeline import AIPipeline
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger
from src.video.video_processor import VideoProcessor


def demo_ai_features(config_manager):
    """AI機能のデモ"""
    print("\n" + "=" * 60)
    print("AI/ML機能デモ - フェーズ2")
    print("=" * 60)

    try:
        # AI パイプラインの初期化
        print("\nAIパイプライン初期化:")
        print("-" * 40)
        pipeline = AIPipeline(config_manager)
        print("✓ AIパイプライン初期化完了")

        # 検出器の初期化とテスト
        print("\n牌検出器テスト:")
        print("-" * 40)
        detector = TileDetector(config_manager)
        print("✓ 牌検出器初期化完了")

        # ダミー画像での検出テスト
        import numpy as np

        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 検出実行
        detections = detector.detect_tiles(test_image)
        print(f"✓ 検出テスト完了 - 検出数: {len(detections)}")

        # 分類器の初期化とテスト
        print("\n牌分類器テスト:")
        print("-" * 40)
        classifier = TileClassifier(config_manager)
        print("✓ 牌分類器初期化完了")

        # ダミー牌画像での分類テスト
        test_tile = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        classification = classifier.classify_tile(test_tile)
        print(
            f"✓ 分類テスト完了 - 予測牌: {classification.tile_name}, "
            f"信頼度: {classification.confidence:.3f}"
        )

        # バッチ分類テスト
        test_tiles = [test_tile for _ in range(3)]
        batch_results = classifier.classify_tiles_batch(test_tiles)
        print(f"✓ バッチ分類テスト完了 - 処理数: {len(batch_results)}")

        # パイプライン統合テスト
        print("\nパイプライン統合テスト:")
        print("-" * 40)
        result = pipeline.process_frame(test_image, frame_id=1)
        print(f"✓ フレーム処理完了 - 処理時間: {result.processing_time:.3f}秒")
        print(f"  検出数: {len(result.detections)}")
        print(f"  分類数: {len(result.classifications)}")

        # モデル管理テスト
        print("\nモデル管理テスト:")
        print("-" * 40)
        model_manager = ModelManager(config_manager)
        print("✓ モデル管理初期化完了")

        models = model_manager.list_models()
        print(f"✓ 登録済みモデル数: {len(models)}")

        # 統計情報表示
        print("\nパイプライン統計:")
        print("-" * 40)
        stats = pipeline.get_statistics()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")

        print("\n✓ AI機能デモ完了")

    except Exception as e:
        print(f"AI機能デモでエラーが発生しました: {e}")
        import traceback

        traceback.print_exc()


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="麻雀牌譜作成システム デモ")
    parser.add_argument("video_path", nargs="?", help="処理する動画ファイルのパス")
    parser.add_argument("--output-dir", help="出力ディレクトリ（省略時は自動設定）")
    parser.add_argument("--fps", type=float, help="フレーム抽出FPS（省略時は設定ファイルの値）")
    parser.add_argument("--scene-detection", action="store_true", help="シーン変更検出を実行")
    parser.add_argument("--ai-demo", action="store_true", help="AI機能のデモを実行")
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="all", help="実行するフェーズ"
    )

    args = parser.parse_args()

    # 設定とログの初期化
    config_manager = ConfigManager()
    setup_logger(config_manager)

    print("=" * 60)
    print("麻雀牌譜作成システム デモ")
    print("=" * 60)

    # AI機能のみのデモ実行
    if args.ai_demo:
        demo_ai_features(config_manager)
        return 0

    # 動画ファイルが指定されていない場合
    if not args.video_path:
        print("動画ファイルが指定されていません。")
        print("AI機能のみをテストする場合は --ai-demo オプションを使用してください。")
        print("例: python demo.py --ai-demo")
        return 1

    # 動画ファイルの存在確認
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return 1

    try:
        # フェーズ1: 基本動画処理
        if args.phase in ["1", "all"]:
            print("\n" + "=" * 60)
            print("フェーズ1: 基本動画処理")
            print("=" * 60)

            # VideoProcessorの初期化
            processor = VideoProcessor(config_manager)

            # FPS設定の上書き
            if args.fps:
                processor.fps = args.fps
                print(f"フレーム抽出FPS: {args.fps}")

            # 動画情報の表示
            print("\n動画情報:")
            print("-" * 40)
            video_info = processor.get_video_info(str(video_path))
            for key, value in video_info.items():
                if key == "file_size":
                    print(f"{key}: {value / (1024 * 1024):.1f} MB")
                elif key == "duration":
                    print(f"{key}: {value:.2f} 秒")
                else:
                    print(f"{key}: {value}")

            # フレーム抽出
            print("\nフレーム抽出:")
            print("-" * 40)
            extracted_frames = processor.extract_frames(str(video_path), args.output_dir)

            print(f"抽出されたフレーム数: {len(extracted_frames)}")
            if extracted_frames:
                print(f"出力ディレクトリ: {Path(extracted_frames[0]).parent}")
                print(f"最初のフレーム: {Path(extracted_frames[0]).name}")
                print(f"最後のフレーム: {Path(extracted_frames[-1]).name}")

            # 関連フレームフィルタリング
            print("\nフレームフィルタリング:")
            print("-" * 40)
            filtered_frames = processor.filter_relevant_frames(extracted_frames)
            print(f"フィルタリング後フレーム数: {len(filtered_frames)}")
            filter_ratio = (
                len(filtered_frames) / len(extracted_frames) * 100 if extracted_frames else 0
            )
            print(f"フィルタリング率: {filter_ratio:.1f}%")

            # シーン変更検出（オプション）
            if args.scene_detection:
                print("\nシーン変更検出:")
                print("-" * 40)
                scene_changes = processor.detect_scene_changes(str(video_path))
                print(f"検出されたシーン変更: {len(scene_changes)}箇所")
                if scene_changes:
                    print("シーン変更タイムスタンプ:")
                    for i, timestamp in enumerate(scene_changes[:10]):  # 最初の10個のみ表示
                        print(f"  {i + 1}: {timestamp:.2f}秒")
                    if len(scene_changes) > 10:
                        print(f"  ... 他{len(scene_changes) - 10}箇所")

        # フェーズ2: AI/ML機能
        if args.phase in ["2", "all"]:
            demo_ai_features(config_manager)

            # 抽出されたフレームがある場合、AI処理も実行
            if args.phase == "all" and "extracted_frames" in locals() and extracted_frames:
                print("\n実際のフレームでのAI処理テスト:")
                print("-" * 40)

                # AIパイプライン初期化
                pipeline = AIPipeline(config_manager)

                # 最初の数フレームで処理テスト
                import cv2

                test_frames = []
                for frame_path in extracted_frames[:3]:  # 最初の3フレーム
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        test_frames.append(frame)

                if test_frames:
                    print(f"テストフレーム数: {len(test_frames)}")

                    # バッチ処理実行
                    results = pipeline.process_frames_batch(test_frames)

                    print("処理結果:")
                    for i, result in enumerate(results):
                        print(
                            f"  フレーム{i + 1}: 検出数={len(result.detections)}, "
                            f"分類数={len(result.classifications)}, "
                            f"処理時間={result.processing_time:.3f}秒"
                        )

                    # 統計表示
                    processed_stats = pipeline.post_process_results(results)
                    if processed_stats:
                        summary = processed_stats.get("summary", {})
                        print("\n統計情報:")
                        print(f"  平均処理時間: {summary.get('average_processing_time', 0):.3f}秒")
                        print(f"  総検出数: {summary.get('total_detections', 0)}")
                        print(f"  総分類数: {summary.get('total_classifications', 0)}")

        print("\n" + "=" * 60)
        print("デモ実行完了")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
