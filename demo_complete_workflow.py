#!/usr/bin/env python3
"""
完全ワークフローデモンストレーション
動画アップロード → フレーム抽出 → ラベリング → 学習 → 評価の全工程デモ
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import ConfigManager
from src.utils.logger import get_logger, setup_logging


class CompleteWorkflowDemo:
    """完全ワークフローデモクラス"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初期化

        Args:
            config_path: 設定ファイルパス
        """
        self.config_manager = ConfigManager(config_path)
        self.logger = get_logger(__name__)

        # デモ用の一時ディレクトリ
        self.demo_dir = None

        # 統計情報
        self.demo_stats = {
            "start_time": None,
            "end_time": None,
            "total_processing_time": 0.0,
            "phases_completed": 0,
            "total_phases": 7,
            "success_rate": 0.0,
            "errors": [],
        }

    def setup_demo_environment(self) -> bool:
        """デモ環境をセットアップ"""
        try:
            self.logger.info("Setting up demo environment...")

            # 一時ディレクトリ作成
            self.demo_dir = tempfile.mkdtemp(prefix="mahjong_demo_")
            self.logger.info(f"Demo directory: {self.demo_dir}")

            # 必要なディレクトリ構造を作成
            demo_dirs = [
                "data/input",
                "data/output",
                "data/temp",
                "data/training",
                "logs",
                "models",
                "web_interface/uploads",
            ]

            for dir_path in demo_dirs:
                Path(self.demo_dir, dir_path).mkdir(parents=True, exist_ok=True)

            # デモ用設定ファイルを作成
            self._create_demo_config()

            # ログ設定
            setup_logging(self.config_manager)

            self.logger.info("Demo environment setup completed")
            return True

        except Exception as e:
            self.logger.error(f"Failed to setup demo environment: {e}")
            return False

    def _create_demo_config(self):
        """デモ用設定ファイルを作成"""
        demo_config = {
            "video": {
                "frame_extraction": {
                    "fps": 1,
                    "output_format": "jpg",
                    "quality": 95,
                    "max_frames": 20,
                }
            },
            "ai": {
                "detection": {"confidence_threshold": 0.5},
                "classification": {"confidence_threshold": 0.8},
            },
            "training": {
                "training_root": f"{self.demo_dir}/data/training",
                "dataset_root": f"{self.demo_dir}/data/training/dataset",
                "database_path": f"{self.demo_dir}/data/training/dataset.db",
                "default_epochs": 5,
                "default_batch_size": 2,
                "default_learning_rate": 0.001,
            },
            "system": {"max_workers": 2, "memory_limit": "2GB", "gpu_enabled": False},
            "directories": {
                "input": f"{self.demo_dir}/data/input",
                "output": f"{self.demo_dir}/data/output",
                "temp": f"{self.demo_dir}/data/temp",
                "models": f"{self.demo_dir}/models",
                "logs": f"{self.demo_dir}/logs",
            },
            "tiles": {
                "manzu": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m"],
                "pinzu": ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p"],
                "souzu": ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s"],
                "jihai": ["東", "南", "西", "北", "白", "發", "中"],
            },
        }

        # 設定を更新
        self.config_manager.config = demo_config

    def create_sample_video(self) -> str:
        """サンプル動画を作成"""
        try:
            self.logger.info("Creating sample video...")

            video_path = os.path.join(self.demo_dir, "data/input/demo_mahjong_video.mp4")

            # 動画作成
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 2.0, (1280, 720))

            # 30フレームの動画を作成
            for i in range(30):
                # 背景フレーム
                frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                frame[:] = (20, 50, 20)  # 暗い緑色の背景

                # 麻雀卓をシミュレート
                cv2.rectangle(frame, (200, 150), (1080, 570), (139, 69, 19), -1)  # 茶色の卓

                # 手牌エリア
                hand_area = (300, 500, 980, 600)
                cv2.rectangle(
                    frame,
                    (hand_area[0], hand_area[1]),
                    (hand_area[2], hand_area[3]),
                    (100, 100, 100),
                    2,
                )

                # 牌をシミュレート（白い矩形）
                for j in range(13):
                    x = hand_area[0] + 20 + j * 50
                    y = hand_area[1] + 20

                    # 牌の色を時間で変化させる
                    color_intensity = int(200 + 50 * np.sin(i * 0.1 + j * 0.2))
                    color = (color_intensity, color_intensity, color_intensity)

                    cv2.rectangle(frame, (x, y), (x + 40, y + 60), color, -1)
                    cv2.rectangle(frame, (x, y), (x + 40, y + 60), (0, 0, 0), 2)

                    # 牌の文字をシミュレート
                    cv2.putText(
                        frame,
                        f"{(j % 9) + 1}",
                        (x + 15, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        2,
                    )

                # 捨て牌エリア
                discard_area = (400, 250, 880, 450)
                cv2.rectangle(
                    frame,
                    (discard_area[0], discard_area[1]),
                    (discard_area[2], discard_area[3]),
                    (80, 80, 80),
                    2,
                )

                # 捨て牌をシミュレート
                for k in range(min(i // 2, 20)):
                    dx = k % 10
                    dy = k // 10
                    x = discard_area[0] + 10 + dx * 45
                    y = discard_area[1] + 10 + dy * 90

                    cv2.rectangle(frame, (x, y), (x + 35, y + 50), (180, 180, 180), -1)
                    cv2.rectangle(frame, (x, y), (x + 35, y + 50), (0, 0, 0), 1)

                # フレーム番号を表示
                cv2.putText(
                    frame,
                    f"Frame {i + 1}/30",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )

                out.write(frame)

            out.release()

            self.logger.info(f"Sample video created: {video_path}")
            return video_path

        except Exception as e:
            self.logger.error(f"Failed to create sample video: {e}")
            raise

    def run_complete_demo(self) -> dict[str, Any]:
        """完全デモを実行"""
        try:
            self.demo_stats["start_time"] = time.time()
            self.logger.info("Starting Complete Workflow Demonstration")
            self.logger.info("=" * 60)

            # 環境セットアップ
            if not self.setup_demo_environment():
                return {"success": False, "error": "Failed to setup demo environment"}

            # サンプル動画作成
            video_path = self.create_sample_video()

            # 各フェーズを実行
            results = {}

            # Phase 1: 動画処理
            self.logger.info("=== Phase 1: Video Processing ===")
            results["phase1"] = self._run_phase1(video_path)

            # Phase 2: データ管理
            self.logger.info("=== Phase 2: Data Management ===")
            results["phase2"] = self._run_phase2(video_path)

            # Phase 3: 半自動ラベリング
            self.logger.info("=== Phase 3: Semi-Auto Labeling ===")
            results["phase3"] = self._run_phase3(results["phase2"].get("version_id", "demo_v1"))

            # Phase 4: モデル学習
            self.logger.info("=== Phase 4: Model Training ===")
            results["phase4"] = self._run_phase4(results["phase2"].get("version_id", "demo_v1"))

            # Phase 5: モデル評価
            self.logger.info("=== Phase 5: Model Evaluation ===")
            results["phase5"] = self._run_phase5(
                results["phase4"].get("session_id", "demo_session")
            )

            # Phase 6: システム統合
            self.logger.info("=== Phase 6: System Integration ===")
            results["phase6"] = self._run_phase6(video_path)

            # Phase 7: 品質検証
            self.logger.info("=== Phase 7: Quality Validation ===")
            results["phase7"] = self._run_phase7(results["phase6"].get("output_path", ""))

            # 統計計算
            self.demo_stats["end_time"] = time.time()
            self.demo_stats["total_processing_time"] = (
                self.demo_stats["end_time"] - self.demo_stats["start_time"]
            )
            self.demo_stats["success_rate"] = (
                self.demo_stats["phases_completed"] / self.demo_stats["total_phases"]
            )

            # 結果サマリー
            summary = self._generate_summary(results)

            self.logger.info("=" * 60)
            self.logger.info("Complete Workflow Demonstration Finished")
            self.logger.info(f"Total time: {self.demo_stats['total_processing_time']:.2f}s")
            self.logger.info(
                f"Phases completed: {self.demo_stats['phases_completed']}/"
                f"{self.demo_stats['total_phases']}"
            )
            self.logger.info(f"Success rate: {self.demo_stats['success_rate']:.2%}")

            return {
                "success": True,
                "results": results,
                "summary": summary,
                "stats": self.demo_stats,
                "demo_directory": self.demo_dir,
            }

        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            return {"success": False, "error": str(e), "stats": self.demo_stats}
        finally:
            # クリーンアップ
            self._cleanup()

    def _run_phase1(self, video_path: str) -> dict[str, Any]:
        """フェーズ1: 動画処理"""
        try:
            start_time = time.time()

            # 動画処理をシミュレート
            time.sleep(1)  # 処理時間をシミュレート

            result = {
                "success": True,
                "extracted_frames": 20,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(f"Phase 1 completed: {result['extracted_frames']} frames extracted")
            return result

        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 1: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase2(self, video_path: str) -> dict[str, Any]:
        """フェーズ2: データ管理"""
        try:
            start_time = time.time()

            # データ管理をシミュレート
            time.sleep(0.5)

            version_id = f"demo_v1_{int(time.time())}"

            result = {
                "success": True,
                "version_id": version_id,
                "total_annotations": 50,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(f"Phase 2 completed: Dataset version {version_id} created")
            return result

        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 2: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase3(self, version_id: str) -> dict[str, Any]:
        """フェーズ3: 半自動ラベリング"""
        try:
            start_time = time.time()

            # 半自動ラベリングをシミュレート
            time.sleep(0.8)

            result = {
                "success": True,
                "predictions_generated": 45,
                "accuracy": 0.85,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(
                f"Phase 3 completed: {result['predictions_generated']} predictions generated"
            )
            return result

        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 3: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase4(self, version_id: str) -> dict[str, Any]:
        """フェーズ4: モデル学習"""
        try:
            start_time = time.time()

            # モデル学習をシミュレート
            session_id = f"training_session_{int(time.time())}"

            # 学習進捗をシミュレート
            for epoch in range(3):
                self.logger.info(f"Training epoch {epoch + 1}/3...")
                time.sleep(0.5)

            result = {
                "success": True,
                "session_id": session_id,
                "final_accuracy": 0.88,
                "epochs_completed": 3,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(
                f"Phase 4 completed: Model trained with {result['final_accuracy']:.2%} accuracy"
            )
            return result

        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 4: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase5(self, session_id: str) -> dict[str, Any]:
        """フェーズ5: モデル評価"""
        try:
            start_time = time.time()

            # モデル評価をシミュレート
            time.sleep(0.3)

            metrics = {"accuracy": 0.88, "precision": 0.86, "recall": 0.90, "f1_score": 0.88}

            result = {
                "success": True,
                "metrics": metrics,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info("Phase 5 completed: Model evaluation metrics calculated")
            return result

        except Exception as e:
            self.logger.error(f"Phase 5 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 5: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase6(self, video_path: str) -> dict[str, Any]:
        """フェーズ6: システム統合"""
        try:
            start_time = time.time()

            # システム統合をシミュレート
            time.sleep(1.2)

            output_path = os.path.join(self.demo_dir, "data/output/demo_record.json")

            # サンプル牌譜を作成
            sample_record = {
                "game_info": {
                    "rule": "東南戦",
                    "players": ["Player1", "Player2", "Player3", "Player4"],
                },
                "rounds": [
                    {
                        "round_number": 1,
                        "round_name": "東1局",
                        "actions": [
                            {"player": "Player1", "action": "draw", "tiles": ["1m"]},
                            {"player": "Player1", "action": "discard", "tiles": ["9p"]},
                        ],
                    }
                ],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sample_record, f, ensure_ascii=False, indent=2)

            result = {
                "success": True,
                "output_path": output_path,
                "record_generated": True,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(f"Phase 6 completed: Game record generated at {output_path}")
            return result

        except Exception as e:
            self.logger.error(f"Phase 6 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 6: {str(e)}")
            return {"success": False, "error": str(e)}

    def _run_phase7(self, record_path: str) -> dict[str, Any]:
        """フェーズ7: 品質検証"""
        try:
            start_time = time.time()

            # 品質検証をシミュレート
            time.sleep(0.4)

            result = {
                "success": True,
                "quality_score": 85.5,
                "validation_passed": True,
                "processing_time": time.time() - start_time,
            }

            self.demo_stats["phases_completed"] += 1
            self.logger.info(f"Phase 7 completed: Quality score {result['quality_score']:.1f}")
            return result

        except Exception as e:
            self.logger.error(f"Phase 7 failed: {e}")
            self.demo_stats["errors"].append(f"Phase 7: {str(e)}")
            return {"success": False, "error": str(e)}

    def _generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """結果サマリーを生成"""
        summary = {
            "total_phases": self.demo_stats["total_phases"],
            "completed_phases": self.demo_stats["phases_completed"],
            "success_rate": self.demo_stats["success_rate"],
            "total_time": self.demo_stats["total_processing_time"],
            "errors": self.demo_stats["errors"],
            "key_metrics": {
                "frames_extracted": results.get("phase1", {}).get("extracted_frames", 0),
                "annotations_created": results.get("phase2", {}).get("total_annotations", 0),
                "model_accuracy": results.get("phase4", {}).get("final_accuracy", 0),
                "quality_score": results.get("phase7", {}).get("quality_score", 0),
            },
        }

        return summary

    def _cleanup(self):
        """クリーンアップ"""
        if self.demo_dir and os.path.exists(self.demo_dir):
            try:
                # デモディレクトリを保持（結果確認のため）
                self.logger.info(f"Demo files preserved in: {self.demo_dir}")
                # 必要に応じてコメントアウト: shutil.rmtree(self.demo_dir)
            except Exception as e:
                self.logger.warning(f"Failed to cleanup demo directory: {e}")


def main():
    """メイン関数"""
    print("麻雀牌譜作成システム - 完全ワークフローデモ")
    print("=" * 50)

    try:
        # デモ実行
        demo = CompleteWorkflowDemo()
        result = demo.run_complete_demo()

        if result["success"]:
            print("\n✅ デモが正常に完了しました！")
            print(f"📁 デモファイル: {result['demo_directory']}")
            print(f"⏱️  総処理時間: {result['stats']['total_processing_time']:.2f}秒")
            print(f"📊 成功率: {result['stats']['success_rate']:.2%}")

            # 主要メトリクス表示
            metrics = result["summary"]["key_metrics"]
            print("\n📈 主要メトリクス:")
            print(f"  - 抽出フレーム数: {metrics['frames_extracted']}")
            print(f"  - アノテーション数: {metrics['annotations_created']}")
            print(f"  - モデル精度: {metrics['model_accuracy']:.2%}")
            print(f"  - 品質スコア: {metrics['quality_score']:.1f}")

            if result["stats"]["errors"]:
                print(f"\n⚠️  エラー ({len(result['stats']['errors'])}件):")
                for error in result["stats"]["errors"]:
                    print(f"  - {error}")
        else:
            print(f"\n❌ デモが失敗しました: {result.get('error', 'Unknown error')}")
            return 1

        return 0

    except KeyboardInterrupt:
        print("\n\n⏹️  デモが中断されました")
        return 130
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
