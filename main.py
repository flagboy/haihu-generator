#!/usr/bin/env python3
"""
麻雀牌譜作成システム - メインアプリケーション
フェーズ4: 最適化・検証・統合
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.integration.system_integrator import SystemIntegrator
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.pipeline.ai_pipeline import AIPipeline
from src.pipeline.game_pipeline import GamePipeline
from src.utils.config import ConfigManager
from src.utils.logger import get_logger, setup_logging
from src.validation.quality_validator import QualityValidator
from src.video.video_processor import VideoProcessor


class MahjongSystemApp:
    """麻雀牌譜作成システムメインアプリケーション"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = get_logger(__name__)

        # コアコンポーネント
        self.video_processor = None
        self.ai_pipeline = None
        self.game_pipeline = None

        # フェーズ4コンポーネント
        self.performance_optimizer = None
        self.quality_validator = None
        self.system_integrator = None

        # 統計情報
        self.session_stats = {
            "start_time": None,
            "end_time": None,
            "total_videos_processed": 0,
            "total_frames_processed": 0,
            "total_records_generated": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
        }

    def initialize(self) -> bool:
        """システムを初期化"""
        try:
            self.logger.info("Initializing Mahjong System...")

            # ログ設定
            setup_logging(self.config_manager)

            # コアコンポーネントを初期化
            self.video_processor = VideoProcessor(self.config_manager)
            self.ai_pipeline = AIPipeline(self.config_manager)
            self.game_pipeline = GamePipeline()

            # フェーズ4コンポーネントを初期化
            self.performance_optimizer = PerformanceOptimizer(self.config_manager)
            self.quality_validator = QualityValidator(self.config_manager)
            self.system_integrator = SystemIntegrator(
                self.config_manager, self.video_processor, self.ai_pipeline, self.game_pipeline
            )

            # ディレクトリ作成
            self._create_directories()

            # システム最適化
            self.performance_optimizer.optimize_system()

            self.logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False

    def _create_directories(self):
        """必要なディレクトリを作成"""
        directories = self.config_manager.get_config().get("directories", {})

        for _dir_type, dir_path in directories.items():
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {dir_path}")

    def process_video(
        self, video_path: str, output_path: str | None = None, enable_validation: bool = True
    ) -> dict[str, Any]:
        """
        動画を処理して天鳳JSON形式の牌譜を生成

        Args:
            video_path: 入力動画パス
            output_path: 出力パス（Noneの場合は自動生成）
            enable_validation: 品質検証を有効にするか

        Returns:
            処理結果
        """
        start_time = time.time()

        try:
            self.logger.info(f"Processing video: {video_path}")

            # 出力パスを決定（天鳳JSON形式固定）
            if output_path is None:
                output_path = self._generate_output_path(video_path)

            # システム統合処理（天鳳JSON形式固定）
            result = self.system_integrator.process_video_complete(
                video_path=video_path,
                output_path=output_path,
                format_type="tenhou_json",
                enable_optimization=True,
                enable_validation=enable_validation,
            )

            # 統計更新
            processing_time = time.time() - start_time
            self._update_session_stats(result, processing_time)

            # 結果ログ
            self.logger.info(f"Video processing completed in {processing_time:.2f}s")
            self.logger.info(
                f"Success: {result['success']}, Quality Score: {result.get('quality_score', 'N/A')}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Video processing failed: {e}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}

    def batch_process(
        self, input_directory: str, output_directory: str, max_workers: int = None
    ) -> dict[str, Any]:
        """
        バッチ処理（天鳳JSON形式固定）

        Args:
            input_directory: 入力ディレクトリ
            output_directory: 出力ディレクトリ
            max_workers: 最大並列数

        Returns:
            バッチ処理結果
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting batch processing: {input_directory}")

            # 動画ファイルを検索
            video_files = self._find_video_files(input_directory)

            if not video_files:
                self.logger.warning("No video files found")
                return {"success": False, "error": "No video files found"}

            # バッチ処理実行（天鳳JSON形式固定）
            batch_result = self.system_integrator.process_batch(
                video_files=video_files,
                output_directory=output_directory,
                format_type="tenhou_json",
                max_workers=max_workers,
            )

            # 統計更新
            processing_time = time.time() - start_time
            batch_result["total_processing_time"] = processing_time

            self.logger.info(f"Batch processing completed in {processing_time:.2f}s")
            self.logger.info(
                f"Processed {len(video_files)} videos, "
                f"Success rate: {batch_result.get('success_rate', 0):.2f}"
            )

            return batch_result

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return {"success": False, "error": str(e), "processing_time": time.time() - start_time}

    def validate_record(self, record_path: str) -> dict[str, Any]:
        """
        牌譜の品質検証

        Args:
            record_path: 牌譜ファイルパス

        Returns:
            検証結果
        """
        try:
            self.logger.info(f"Validating record: {record_path}")

            validation_result = self.quality_validator.validate_record_file(record_path)

            self.logger.info(
                f"Validation completed. Score: {validation_result.get('overall_score', 'N/A')}"
            )

            return validation_result

        except Exception as e:
            self.logger.error(f"Record validation failed: {e}")
            return {"success": False, "error": str(e)}

    def optimize_performance(self) -> dict[str, Any]:
        """パフォーマンス最適化"""
        try:
            self.logger.info("Starting performance optimization...")

            optimization_result = self.performance_optimizer.optimize_full_system()

            self.logger.info("Performance optimization completed")

            return optimization_result

        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")
            return {"success": False, "error": str(e)}

    def get_system_status(self) -> dict[str, Any]:
        """システム状態を取得"""
        try:
            status = {
                "system_info": self.system_integrator.get_system_info(),
                "performance_metrics": self.performance_optimizer.get_current_metrics(),
                "session_statistics": self.session_stats.copy(),
                "component_status": {
                    "video_processor": self.video_processor is not None,
                    "ai_pipeline": self.ai_pipeline is not None,
                    "game_pipeline": self.game_pipeline is not None,
                    "performance_optimizer": self.performance_optimizer is not None,
                    "quality_validator": self.quality_validator is not None,
                    "system_integrator": self.system_integrator is not None,
                },
            }

            return status

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"error": str(e)}

    def _generate_output_path(self, video_path: str) -> str:
        """天鳳JSON形式の出力パスを生成"""
        video_name = Path(video_path).stem
        output_dir = self.config_manager.get_config()["directories"]["output"]

        # 天鳳JSON形式固定
        extension = ".json"

        return os.path.join(output_dir, f"{video_name}_tenhou_record{extension}")

    def _find_video_files(self, directory: str) -> list[str]:
        """ディレクトリから動画ファイルを検索"""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(Path(directory).glob(f"*{ext}"))
            video_files.extend(Path(directory).glob(f"*{ext.upper()}"))

        return [str(f) for f in video_files]

    def _update_session_stats(self, result: dict[str, Any], processing_time: float):
        """セッション統計を更新"""
        if self.session_stats["start_time"] is None:
            self.session_stats["start_time"] = time.time() - processing_time

        self.session_stats["end_time"] = time.time()
        self.session_stats["total_videos_processed"] += 1

        if result.get("success", False):
            self.session_stats["total_records_generated"] += 1

        # 成功率を計算
        self.session_stats["success_rate"] = (
            self.session_stats["total_records_generated"]
            / self.session_stats["total_videos_processed"]
        )

        # 平均処理時間を更新
        total_time = self.session_stats["end_time"] - self.session_stats["start_time"]
        self.session_stats["average_processing_time"] = (
            total_time / self.session_stats["total_videos_processed"]
        )


def create_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーを作成"""
    parser = argparse.ArgumentParser(
        description="麻雀牌譜作成システム - 動画から天鳳JSON形式の牌譜を自動生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 単一動画を処理（天鳳JSON形式で出力）
  python main.py process video.mp4

  # バッチ処理（天鳳JSON形式で出力）
  python main.py batch input_dir output_dir

  # 牌譜検証
  python main.py validate record.json

  # システム最適化
  python main.py optimize

  # システム状態確認
  python main.py status
        """,
    )

    parser.add_argument(
        "--config", "-c", default="config.yaml", help="設定ファイルパス (default: config.yaml)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログを出力")

    subparsers = parser.add_subparsers(dest="command", help="実行コマンド")

    # process コマンド
    process_parser = subparsers.add_parser("process", help="単一動画を処理（天鳳JSON形式）")
    process_parser.add_argument("video_path", help="入力動画ファイルパス")
    process_parser.add_argument("--output", "-o", help="出力ファイルパス")
    process_parser.add_argument("--no-validation", action="store_true", help="品質検証を無効化")

    # batch コマンド
    batch_parser = subparsers.add_parser("batch", help="バッチ処理（天鳳JSON形式）")
    batch_parser.add_argument("input_dir", help="入力ディレクトリ")
    batch_parser.add_argument("output_dir", help="出力ディレクトリ")
    batch_parser.add_argument("--workers", "-w", type=int, help="並列処理数")

    # validate コマンド
    validate_parser = subparsers.add_parser("validate", help="牌譜検証")
    validate_parser.add_argument("record_path", help="牌譜ファイルパス")

    # optimize コマンド
    subparsers.add_parser("optimize", help="システム最適化")

    # status コマンド
    subparsers.add_parser("status", help="システム状態確認")

    return parser


def main() -> int:
    """メイン関数"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # アプリケーション初期化
    app = MahjongSystemApp(args.config)

    if not app.initialize():
        print("システムの初期化に失敗しました", file=sys.stderr)
        return 1

    try:
        # コマンド実行
        if args.command == "process":
            result = app.process_video(
                video_path=args.video_path,
                output_path=args.output,
                enable_validation=not args.no_validation,
            )

            if result["success"]:
                print(f"処理完了: {result.get('output_path', 'N/A')}")
                print(f"品質スコア: {result.get('quality_score', 'N/A')}")
                return 0
            else:
                print(f"処理失敗: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1

        elif args.command == "batch":
            result = app.batch_process(
                input_directory=args.input_dir,
                output_directory=args.output_dir,
                max_workers=args.workers,
            )

            if result["success"]:
                print("バッチ処理完了")
                print(f"成功率: {result.get('success_rate', 0):.2f}")
                print(f"処理時間: {result.get('total_processing_time', 0):.2f}秒")
                return 0
            else:
                print(f"バッチ処理失敗: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1

        elif args.command == "validate":
            result = app.validate_record(args.record_path)

            if result.get("success", False):
                print("検証完了")
                print(f"総合スコア: {result.get('overall_score', 'N/A')}")
                print(
                    f"詳細: {json.dumps(result.get('details', {}), indent=2, ensure_ascii=False)}"
                )
                return 0
            else:
                print(f"検証失敗: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1

        elif args.command == "optimize":
            result = app.optimize_performance()

            if result.get("success", False):
                print("最適化完了")
                print(f"詳細: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return 0
            else:
                print(f"最適化失敗: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return 1

        elif args.command == "status":
            status = app.get_system_status()
            print("システム状態:")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0

        else:
            parser.print_help()
            return 1

    except KeyboardInterrupt:
        print("\n処理が中断されました")
        return 130

    except Exception as e:
        print(f"予期しないエラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
